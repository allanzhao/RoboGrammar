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

  // Create a boxy snake
  std::shared_ptr<Robot> robot = std::make_shared<Robot>();
  for (Index i = 0; i < 3; ++i) {
    robot->joints_.push_back({
        i - 1,
        (i == 0) ? JointType::FREE : JointType::HINGE,
        Vector3{0, 0, 0},
        Vector4{0, 0, 0, 1}});
    robot->bodies_.push_back({
        i,
        Vector3{0, -0.5, 0},
        Vector4{0, 0, 0, 1},
        1.0,
        Vector3{0.0833, 0, 0.0833}});
    robot->shapes_.push_back({
        i,
        ShapeType::BOX,
        Vector3{0, 0, 0},
        Vector4{0, 0, 0, 1},
        Vector3{0.1, 0.5, 0.1}});
  }

  std::shared_ptr<BulletSimulation> sim = std::make_shared<BulletSimulation>();
  sim->addRobot(robot);
  GLFWRenderer renderer;
  renderer.render(*sim);
}

