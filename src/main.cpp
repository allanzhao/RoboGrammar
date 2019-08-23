#include <args.hxx>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <random>
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
  constexpr int horizon = 240 * 4;
  // Use the provided random seed to generate all other seeds
  std::mt19937 generator(args::get(seed_flag));
  torch::Device device(args::get(device_flag));

  // Set Torch random seed
  torch::manual_seed(generator());

  // Create a quadruped robot
  std::shared_ptr<Robot> robot = std::make_shared<Robot>(
      /*link_density=*/1.0,
      /*link_radius=*/0.05,
      /*friction=*/0.9,
      /*motor_kp=*/2.0,
      /*motor_kd=*/0.1);
  robot->links_.emplace_back(
      /*parent=*/-1,
      /*joint_type=*/JointType::FREE,
      /*joint_pos=*/1.0,
      /*joint_rot=*/Quaternion::Identity(),
      /*joint_axis=*/Vector3{0.0, 0.0, 1.0},
      /*length=*/0.4);
  for (Index i = 0; i < 4; ++i) {
    Quaternion leg_rot(Eigen::AngleAxis<Scalar>((i - 0.5) * M_PI / 2, Vector3::UnitY()));
    Quaternion thigh_rot(Eigen::AngleAxis<Scalar>(M_PI / 2, Vector3::UnitZ()) *
        Eigen::AngleAxis<Scalar>((i - 0.5) * -M_PI / 2, Vector3::UnitY()));
    Quaternion shin_rot(Eigen::AngleAxis<Scalar>(0.0, Vector3::UnitZ()));
    robot->links_.emplace_back(
        /*parent=*/0,
        /*joint_type=*/JointType::FIXED,
        /*joint_pos=*/(i < 2) ? 1.0 : 0.0,
        /*joint_rot=*/leg_rot,
        /*joint_axis=*/Vector3{0.0, 1.0, 0.0},
        /*length=*/0.2);
    robot->links_.emplace_back(
        /*parent=*/i * 3 + 1,
        /*joint_type=*/JointType::HINGE,
        /*joint_pos=*/1.0,
        /*joint_rot=*/thigh_rot,
        /*joint_axis=*/Vector3{0.0, 0.0, 1.0},
        /*length=*/0.2);
    robot->links_.emplace_back(
        /*parent=*/i * 3 + 2,
        /*joint_type=*/JointType::HINGE,
        /*joint_pos=*/1.0,
        /*joint_rot=*/shin_rot,
        /*joint_axis=*/Vector3{0.0, 0.0, 1.0},
        /*length=*/0.2);
  }

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
  auto objective_fn = [&](const Simulation &sim, int step) -> Scalar {
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
  unsigned int opt_seed = generator();
  auto value_estimator = std::make_shared<FCValueEstimator>(
      *main_sim, /*robot_idx=*/0, /*device=*/device, /*batch_size=*/64,
      /*epoch_count=*/3);
  MPPIOptimizer optimizer(
      /*kappa=*/100.0, /*dof_count=*/dof_count, /*horizon=*/horizon,
      /*sample_count=*/64, /*thread_count=*/thread_count, /*seed=*/opt_seed,
      /*make_sim_fn=*/make_sim_fn, /*objective_fn=*/objective_fn,
      /*value_estimator=*/value_estimator);
  for (int i = 0; i < 20; ++i) {
    optimizer.update();
  }
  // Receding horizon control
  constexpr int interval = 4;
  MatrixX input_sequence = MatrixX::Zero(dof_count, 240 * interval);
  for (int j = 0; j < input_sequence.cols(); j += interval) {
    optimizer.update();
    input_sequence.block(0, j, dof_count, interval) = optimizer.input_sequence_.leftCols(interval);
    optimizer.advance(interval);
  }

  main_sim->saveState();
  GLFWRenderer renderer;
  double sim_time = glfwGetTime();
  int j = 0;
  while (!renderer.shouldClose()) {
    double current_time = glfwGetTime();
    while (sim_time < current_time) {
      main_sim->setJointTargetPositions(robot_idx, input_sequence.col(j++));
      main_sim->step();
      renderer.update(time_step);
      sim_time += time_step;
      if (j >= input_sequence.cols()) {
        j = 0;
        main_sim->restoreState();
      }
    }
    renderer.render(*main_sim);
  }
}
