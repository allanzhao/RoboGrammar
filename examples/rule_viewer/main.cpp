#include <args.hxx>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <lodepng.h>
#include <memory>
#include <robot_design/glfw_viewer.h>
#include <robot_design/graph.h>
#include <robot_design/render.h>
#include <robot_design/robot.h>
#include <robot_design/sim.h>
#include <robot_design/types.h>
#include <unordered_set>
#include <vector>

using namespace robot_design;

enum class RuleSide : unsigned int { LHS, RHS };

int main(int argc, char **argv) {
  args::ArgumentParser parser("Robot design rule viewer.");
  args::HelpFlag help_flag(parser, "help", "Display this help message",
                           {'h', "help"});
  args::Positional<std::string> graph_file_arg(
      parser, "graph_file", "Graph file (.dot)", args::Options::Required);
  args::Positional<unsigned int> rule_arg(parser, "rule", "Rule index",
                                          args::Options::Required);
  args::MapPositional<std::string, RuleSide> side_arg(
      parser, "side", "Rule side (lhs|rhs)",
      {{"lhs", RuleSide::LHS}, {"rhs", RuleSide::RHS}}, RuleSide::LHS,
      args::Options::Required);
  args::Flag render_flag(parser, "render", "Render to a window",
                         {'r', "render"});
  args::ValueFlag<std::string> save_image_flag(
      parser, "save_image", "Save PNG image to file", {"save_image"});

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

  // Load rule graphs
  std::vector<Graph> rule_graphs = loadGraphs(args::get(graph_file_arg));
  if (rule_graphs.empty()) {
    std::cerr << "Graph file does not contain any graphs" << std::endl;
    return 1;
  }
  std::cout << "Number of graphs: " << rule_graphs.size() << std::endl;

  // Convert graphs to rules
  std::vector<Rule> rules;
  for (const Graph &rule_graph : rule_graphs) {
    rules.push_back(createRuleFromGraph(rule_graph));
  }

  // Generate a robot graph from the selected rule
  const Rule &rule = rules.at(args::get(rule_arg));
  Graph robot_graph;
  if (args::get(side_arg) == RuleSide::LHS) {
    robot_graph = rule.lhs_;
  } else { // RuleSide::RHS
    robot_graph = rule.rhs_;
  }
  assert(!robot_graph.nodes_.empty());
  // Provide placeholder attributes for nonterminal nodes
  for (Node &node : robot_graph.nodes_) {
    if (node.attrs_.shape_ == LinkShape::NONE) {
      node.attrs_.shape_ = LinkShape::CAPSULE;
      node.attrs_.length_ = 0.2;
      node.attrs_.radius_ = 0.025;
    }
  }
  // Provide placeholder attributes for nonterminal edges
  for (Edge &edge : robot_graph.edges_) {
    if (edge.attrs_.joint_type_ == JointType::NONE) {
      edge.attrs_.joint_type_ = JointType::FIXED;
    }
  }
  // If LHS, display require_label_ instead of label_
  if (args::get(side_arg) == RuleSide::LHS) {
    for (Node &node : robot_graph.nodes_) {
      node.attrs_.label_ = node.attrs_.require_label_;
    }
    for (Edge &edge : robot_graph.edges_) {
      edge.attrs_.label_ = edge.attrs_.require_label_;
    }
  }
  // Color nodes and edges which are added or removed
  std::unordered_set<NodeIndex> nodes_in_common;
  std::unordered_set<EdgeIndex> edges_in_common;
  Color diff_color;
  if (args::get(side_arg) == RuleSide::LHS) {
    // Nodes and edges are removed in the LHS
    nodes_in_common.insert(rule.common_to_lhs_.node_mapping_.begin(),
                           rule.common_to_lhs_.node_mapping_.end());
    for (const auto &edges : rule.common_to_lhs_.edge_mapping_) {
      edges_in_common.insert(edges.begin(), edges.end());
    }
    diff_color = {0.5f, 0.0f, 0.0f}; // Maroon
  } else {                           // RuleSide::RHS
    // Nodes and edges are added in the RHS
    nodes_in_common.insert(rule.common_to_rhs_.node_mapping_.begin(),
                           rule.common_to_rhs_.node_mapping_.end());
    for (const auto &edges : rule.common_to_rhs_.edge_mapping_) {
      edges_in_common.insert(edges.begin(), edges.end());
    }
    diff_color = {0.0f, 0.5f, 0.0f}; // Green
  }
  for (NodeIndex i = 0; i < robot_graph.nodes_.size(); ++i) {
    if (nodes_in_common.count(i) == 0) {
      // Node is not in the common graph
      robot_graph.nodes_[i].attrs_.color_ = diff_color;
    }
  }
  for (EdgeIndex m = 0; m < robot_graph.edges_.size(); ++m) {
    if (edges_in_common.count(m) == 0) {
      // Edge is not in the common graph
      robot_graph.edges_[m].attrs_.color_ = diff_color;
    }
  }
  auto robot = std::make_shared<Robot>(buildRobot(robot_graph));

  // Find an initial offset that places the robot in the center of the view
  Vector3 offset;
  {
    BulletSimulation temp_sim(time_step);
    temp_sim.addRobot(robot, Vector3::Zero(), Quaternion::Identity());
    Vector3 lower, upper;
    temp_sim.getRobotWorldAABB(temp_sim.findRobotIndex(*robot), lower, upper);
    offset = -0.5 * (lower + upper);
  }

  // Define a lambda function for making simulation instances
  auto make_sim_fn = [&]() -> std::shared_ptr<Simulation> {
    std::shared_ptr<BulletSimulation> sim =
        std::make_shared<BulletSimulation>(time_step);
    sim->addRobot(robot, offset, Quaternion::Identity());
    return sim;
  };

  // Create the "main" simulation
  std::shared_ptr<Simulation> main_sim = make_sim_fn();

  std::string save_image_path = args::get(save_image_flag);
  if (!save_image_path.empty()) {
    GLFWViewer viewer(/*hidden=*/false);
    viewer.camera_params_.distance_ = 1.0;
    viewer.update(time_step);
    int fb_width, fb_height;
    viewer.getFramebufferSize(fb_width, fb_height);
    std::unique_ptr<unsigned char[]> rgba(
        new unsigned char[4 * fb_width * fb_height]);
    viewer.render(*main_sim, rgba.get());
    std::unique_ptr<unsigned char[]> rgba_flipped(
        new unsigned char[4 * fb_width * fb_height]);
    for (int i = 0; i < fb_height; ++i) {
      std::memcpy(&rgba_flipped[i * fb_width * 4],
                  &rgba[(fb_height - i - 1) * fb_width * 4], fb_width * 4);
    }
    unsigned int error = lodepng::encode(save_image_path, rgba_flipped.get(),
                                         fb_width, fb_height);
    if (error) {
      std::cerr << "Failed to save image: " << lodepng_error_text(error)
                << std::endl;
    }
  }

  if (args::get(render_flag)) {
    // View in a window
    GLFWViewer viewer;
    viewer.camera_params_.distance_ = 1.0;
    double sim_time = glfwGetTime();
    while (!viewer.shouldClose()) {
      double current_time = glfwGetTime();
      while (sim_time < current_time) {
        viewer.update(time_step);
        sim_time += time_step;
      }
      viewer.render(*main_sim);
    }
  }
}
