#include <args.hxx>
#include <iostream>
#include <robot_design/grammar.h>
#include <robot_design/render.h>
#include <robot_design/robot.h>
#include <robot_design/sim.h>

using namespace robot_design;

int main(int argc, char **argv) {
  args::ArgumentParser parser("Robot design rule viewer.");
  args::HelpFlag help_flag(
      parser, "help", "Display this help message", {'h', "help"});
  args::Positional<std::string> rule_file_arg(
      parser, "rule_file", "Rule file (.dot)", args::Options::Required);

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

  std::shared_ptr<Rule> rule = loadRule(args::get(rule_file_arg));

  //// Create a floor
  //std::shared_ptr<Prop> floor = std::make_shared<Prop>(
  //    /*density=*/0.0,  // static
  //    /*friction=*/0.9,
  //    /*half_extents=*/Vector3{10.0, 1.0, 10.0});

  //constexpr Scalar time_step = 1.0 / 240;
  //auto sim = std::make_shared<BulletSimulation>(time_step);
  //sim->addProp(floor, Vector3{0.0, -1.0, 0.0}, Quaternion::Identity());
  //sim->addRobot(robot, Vector3{0.0, 0.45, 0.0}, Quaternion::Identity());
  //Index robot_idx = sim->findRobotIndex(*robot);
  //int dof_count = sim->getRobotDofCount(robot_idx);

  //GLFWRenderer renderer;
  //double last_time = glfwGetTime();
  //while (!renderer.shouldClose()) {
  //  sim->setJointTargetPositions(robot_idx, VectorX::Zero(dof_count));
  //  sim->step();
  //  double current_time = glfwGetTime();
  //  renderer.update(current_time - last_time);
  //  last_time = current_time;
  //  renderer.render(*sim);
  //}
}
