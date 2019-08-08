#include <args.hxx>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <robot_design/control.h>
#include <robot_design/render.h>
#include <robot_design/sim.h>

using namespace robot_design;

int main(int argc, char **argv) {
  args::ArgumentParser parser("Robot design search demo.");
  args::HelpFlag help(parser, "help", "Display this help message",
                      {'h', "help"});
  args::Flag verbose(parser, "verbose", "Enable verbose mode",
                     {'v', "verbose"});

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

  // Create a quadruped robot
  std::shared_ptr<Robot> robot = std::make_shared<Robot>(
      /*link_density=*/1.0,
      /*link_radius=*/0.05,
      /*friction=*/0.9);
  robot->links_.emplace_back(
      /*parent=*/-1,
      /*joint_type=*/JointType::FREE,
      /*joint_pos=*/1.0,
      /*joint_rot=*/Quaternion::Identity(),
      /*joint_axis=*/Vector3{0.0, 0.0, 1.0},
      /*length=*/0.4);
  for (Index i = 0; i < 4; ++i) {
    Quaternion joint_rot(Eigen::AngleAxis<Scalar>(0.0, Vector3::UnitZ()) *
                         Eigen::AngleAxis<Scalar>((i - 0.5) * M_PI / 2, Vector3::UnitY()) *
                         Eigen::AngleAxis<Scalar>(0.0, Vector3::UnitX()));
    robot->links_.emplace_back(
        /*parent=*/0,
        /*joint_type=*/JointType::HINGE,
        /*joint_pos=*/(i < 2) ? 1.0 : 0.0,
        /*joint_rot=*/joint_rot,
        /*joint_axis=*/Vector3{0.0, 1.0, 0.0},
        /*length=*/0.2);
    robot->links_.emplace_back(
        /*parent=*/i * 2 + 1,
        /*joint_type=*/JointType::HINGE,
        /*joint_pos=*/1.0,
        /*joint_rot=*/Quaternion::Identity(),
        /*joint_axis=*/Vector3{0.0, 0.0, 1.0},
        /*length=*/0.4);
  }

  // Create a floor
  std::shared_ptr<Prop> floor = std::make_shared<Prop>(
      /*density=*/0.0,  // static
      /*friction=*/0.9,
      /*half_extents=*/Vector3{10.0, 1.0, 10.0});

  std::shared_ptr<BulletSimulation> sim = std::make_shared<BulletSimulation>();
  sim->addProp(floor, Vector3{0.0, -1.0, 0.0}, Quaternion::Identity());
  sim->addRobot(robot, Vector3{0.0, 1.0, 0.0}, Quaternion::Identity());
  PDController controller(*robot, *sim);
  controller.kp_.fill(10.0);
  controller.kd_.fill(1.0);
  controller.pos_target_ << 0.0, -M_PI / 2, 0.0, -M_PI / 2, 0.0, -M_PI / 2, 0.0, -M_PI / 2;
  GLFWRenderer renderer;

  const double time_step = 1.0 / 240;
  double sim_time = glfwGetTime();
  while (!renderer.shouldClose()) {
    double current_time = glfwGetTime();
    while (sim_time < current_time) {
      controller.update();
      sim->step(time_step);
      renderer.update(time_step);
      sim_time += time_step;
    }
    renderer.render(*sim);
  }
}

