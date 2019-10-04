#include <args.hxx>
#include <iostream>
#include <robot_design/render.h>
#include <robot_design/robot.h>
#include <robot_design/sim.h>

using namespace robot_design;

int main(int argc, char **argv) {
  args::ArgumentParser parser("Robot design viewer.");
  args::HelpFlag help_flag(
      parser, "help", "Display this help message", {'h', "help"});

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
      /*friction=*/0.9,
      /*motor_kp=*/2.0,
      /*motor_kd=*/0.1);
  robot->links_.emplace_back(
      /*parent=*/-1,
      /*joint_type=*/JointType::FREE,
      /*joint_pos=*/1.0,
      /*joint_rot=*/Quaternion::Identity(),
      /*joint_axis=*/Vector3{0.0, 0.0, 1.0},
      /*shape=*/LinkShape::CAPSULE,
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
        /*shape=*/LinkShape::CAPSULE,
        /*length=*/0.2);
    robot->links_.emplace_back(
        /*parent=*/i * 3 + 1,
        /*joint_type=*/JointType::HINGE,
        /*joint_pos=*/1.0,
        /*joint_rot=*/thigh_rot,
        /*joint_axis=*/Vector3{0.0, 0.0, 1.0},
        /*shape=*/LinkShape::CAPSULE,
        /*length=*/0.2);
    robot->links_.emplace_back(
        /*parent=*/i * 3 + 2,
        /*joint_type=*/JointType::HINGE,
        /*joint_pos=*/1.0,
        /*joint_rot=*/shin_rot,
        /*joint_axis=*/Vector3{0.0, 0.0, 1.0},
        /*shape=*/LinkShape::CAPSULE,
        /*length=*/0.2);
  }

  // Create a floor
  std::shared_ptr<Prop> floor = std::make_shared<Prop>(
      /*density=*/0.0,  // static
      /*friction=*/0.9,
      /*half_extents=*/Vector3{10.0, 1.0, 10.0});

  constexpr Scalar time_step = 1.0 / 240;
  auto sim = std::make_shared<BulletSimulation>(time_step);
  sim->addProp(floor, Vector3{0.0, -1.0, 0.0}, Quaternion::Identity());
  sim->addRobot(robot, Vector3{0.0, 0.45, 0.0}, Quaternion::Identity());
  Index robot_idx = sim->findRobotIndex(*robot);
  int dof_count = sim->getRobotDofCount(robot_idx);

  GLFWRenderer renderer;
  double last_time = glfwGetTime();
  while (!renderer.shouldClose()) {
    sim->setJointTargetPositions(robot_idx, VectorX::Zero(dof_count));
    sim->step();
    double current_time = glfwGetTime();
    renderer.update(current_time - last_time);
    last_time = current_time;
    renderer.render(*sim);
  }
}
