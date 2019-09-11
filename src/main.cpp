#include <args.hxx>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <random>
#include <robot_design/articulated_robot.h>
#include <robot_design/design.h>
#include <robot_design/grammar.h>
#include <robot_design/optim.h>
#include <robot_design/render.h>
#include <robot_design/sim.h>
#include <thread>
#include <torch/torch.h>

using namespace robot_design;

int main(int argc, char **argv) {
  args::ArgumentParser parser("Robot design search demo.");
  args::HelpFlag help_flag(parser, "help", "Display this help message",
                           {'h', "help"});
  args::Flag verbose_flag(parser, "verbose", "Enable verbose mode",
                          {'v', "verbose"});
  args::ValueFlag<unsigned int> seed_flag(parser, "seed", "Random seed",
                                          {'s', "seed"});
  args::MapFlag<std::string, torch::DeviceType> device_flag(
      parser, "device", "Torch device (cpu|cuda)", {'d', "device"},
      {{"cpu", torch::kCPU}, {"cuda", torch::kCUDA}}, torch::kCPU);

  // Don't show the (overly verbose) message about the '--' flag
  parser.helpParams.showTerminator = false;

  try {
    parser.ParseCLI(argc, argv);
  } catch (const args::Completion &e) {
    std::cout << e.what();
    return 0;
  } catch (const args::Help &) {
    std::cout << parser;
    return 0;
  } catch (const args::ParseError &e) {
    std::cerr << e.what() << std::endl;
    std::cerr << parser;
    return 1;
  } catch (const args::RequiredError &e) {
    std::cerr << e.what() << std::endl;
    std::cerr << parser;
    return 1;
  }

  constexpr Scalar time_step = 1.0 / 240;
  constexpr int interval = 4;
  constexpr int horizon = 64;
  constexpr Scalar discount_factor = 0.99;
  // Use the provided random seed to generate all other seeds
  std::mt19937 generator(args::get(seed_flag));
  torch::Device device(args::get(device_flag));

  // Set Torch random seed
  torch::manual_seed(generator());

  // Sample a robot design
  auto ar_grammar = std::make_shared<ArticulatedRobotGrammar>();
  DesignSampler sampler(ar_grammar, generator());
  Design design = sampler.sampleDesign(ar_grammar->robot_);
  for (Rule rule : design.derivation_) {
    for (Symbol symbol : ar_grammar->rule_defs_[rule].rhs_) {
      std::cout << ar_grammar->symbol_defs_[symbol].name_ << std::endl;
    }
  }
  std::shared_ptr<Robot> robot = buildArticulatedRobot(design, *ar_grammar);

  // Create a floor
  std::shared_ptr<Prop> floor = std::make_shared<Prop>(
      /*density=*/0.0,  // static
      /*friction=*/0.9,
      /*half_extents=*/Vector3{10.0, 1.0, 10.0});

  // Define a lambda function for making simulation instances
  auto make_sim_fn = [&]() -> std::shared_ptr<Simulation> {
    std::shared_ptr<BulletSimulation> sim = std::make_shared<BulletSimulation>(time_step);
    sim->addProp(floor, Vector3{0.0, -1.0, 0.0}, Quaternion::Identity());
    sim->addRobot(robot, Vector3{0.0, 0.45, 0.0}, Quaternion::Identity());
    return sim;
  };

  // Define an objective function
  auto objective_fn = [&](const Simulation &sim) -> Scalar {
    Index robot_idx = sim.findRobotIndex(*robot);
    Vector6 base_vel;
    sim.getLinkVelocity(robot_idx, 0, base_vel);
    Scalar forward_progress_term = base_vel(3) * time_step;
    return 1.0 * forward_progress_term;
  };

  // Create the "main" simulation
  std::shared_ptr<Simulation> main_sim = make_sim_fn();
  Index robot_idx = main_sim->findRobotIndex(*robot);
  int dof_count = main_sim->getRobotDofCount(robot_idx);
  unsigned int thread_count = std::thread::hardware_concurrency();
  auto value_estimator = std::make_shared<FCValueEstimator>(
      *main_sim, /*robot_idx=*/robot_idx, /*device=*/device, /*batch_size=*/64,
      /*epoch_count=*/3);
  int episode_len = 250;
  MatrixX input_sequence(dof_count, episode_len);
  MatrixX obs(value_estimator->getObservationSize(), episode_len + 1);
  VectorX rewards(episode_len);
  VectorX returns(episode_len + 1);

  for (int episode_idx = 0; episode_idx < 3; ++episode_idx) {
    std::cout << "Episode " << episode_idx << std::endl;

    unsigned int opt_seed = generator();
    MPPIOptimizer optimizer(
        /*kappa=*/100.0, /*discount_factor=*/discount_factor,
        /*dof_count=*/dof_count, /*interval=*/interval, /*horizon=*/horizon,
        /*sample_count=*/128, /*thread_count=*/thread_count, /*seed=*/opt_seed,
        /*make_sim_fn=*/make_sim_fn, /*objective_fn=*/objective_fn,
        /*value_estimator=*/value_estimator);
    for (int i = 0; i < 10; ++i) {
      optimizer.update();
    }

    // Run the main simulation in lockstep with the optimizer's simulations
    main_sim->saveState();
    for (int j = 0; j < input_sequence.cols(); ++j) {
      optimizer.update();
      input_sequence.col(j) = optimizer.input_sequence_.col(0);
      optimizer.advance(1);

      value_estimator->getObservation(*main_sim, obs.col(j));
      rewards(j) = 0.0;
      for (int i = 0; i < interval; ++i) {
        main_sim->setJointTargetPositions(robot_idx, input_sequence.col(j));
        main_sim->step();
        rewards(j) += objective_fn(*main_sim);
      }
    }
    value_estimator->getObservation(*main_sim, obs.col(episode_len));
    main_sim->restoreState();

    // Bootstrap returns with value estimator
    value_estimator->estimateValue(obs.col(episode_len), returns.tail<1>());
    for (int j = episode_len - 1; j >= 0; --j) {
      returns(j) = rewards(j) + discount_factor * returns(j + 1);
    }
    value_estimator->train(obs.leftCols(episode_len),
                           returns.head(episode_len));

    std::cout << "Total reward: " << rewards.sum() << std::endl;
  }

  main_sim->saveState();
  GLFWRenderer renderer;
  double sim_time = glfwGetTime();
  int i = 0, j = 0;
  while (!renderer.shouldClose()) {
    double current_time = glfwGetTime();
    while (sim_time < current_time) {
      main_sim->setJointTargetPositions(robot_idx, input_sequence.col(j));
      main_sim->step();
      renderer.update(time_step);
      sim_time += time_step;
      ++i;
      if (i >= interval) {
        i = 0;
        ++j;
      }
      if (j >= input_sequence.cols()) {
        i = 0;
        j = 0;
        main_sim->restoreState();
      }
    }
    renderer.render(*main_sim);
  }
}
