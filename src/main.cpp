#include <args.hxx>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
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

  // Create a snake robot
  std::shared_ptr<Robot> robot = std::make_shared<Robot>();
  for (Index i = 0; i < 5; ++i) {
    Quaternion joint_rot(Eigen::AngleAxis<Scalar>(M_PI / 2, Vector3::UnitZ()) *
                         Eigen::AngleAxis<Scalar>(0.0, Vector3::UnitY()) *
                         Eigen::AngleAxis<Scalar>(0.0, Vector3::UnitX()));
    robot->links_.emplace_back(
        /*parent=*/i - 1,
        /*joint_type=*/(i == 0) ? JointType::FREE : JointType::HINGE,
        /*joint_pos=*/1.0,
        /*joint_rot=*/joint_rot,
        /*joint_axis=*/Vector3{0.0, 0.0, 1.0},
        /*length=*/1.0);
  }

  std::shared_ptr<BulletSimulation> sim = std::make_shared<BulletSimulation>(
      /*link_density=*/10.0,
      /*link_radius=*/0.05,
      /*friction=*/0.9);
  sim->addRobot(robot);
  GLFWRenderer renderer;
  renderer.render(*sim);
}

