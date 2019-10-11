#include <deque>
#include <iostream>
#include <map>
#include <memory>
#include <robot_design/graph.h>
#include <robot_design/internal/dot_parsing.h>
#include <robot_design/internal/dot_rules.h>
#include <robot_design/robot.h>
#include <robot_design/types.h>
#include <tao/pegtl.hpp>
#include <tao/pegtl/contrib/parse_tree.hpp>
#include <tao/pegtl/contrib/parse_tree_to_dot.hpp>
#include <vector>

namespace robot_design {

std::shared_ptr<Graph> loadGraph(const std::string &filename) {
  tao::pegtl::file_input<> input(filename);
  auto root = tao::pegtl::parse_tree::parse<
      tao::pegtl::pad<dot_rules::graph, dot_rules::sep>,
      dot_parsing::dot_selector>(input);
  auto graph = std::make_shared<Graph>();

  if (!root || root->children.empty()) { return graph; }
  auto &graph_node = root->children[0];
  graph->name_ = graph_node->children[0]->string();

  std::vector<NodeAttributes> node_attr_stack = {NodeAttributes{
      /*joint_type=*/JointType::HINGE,
      /*joint_axis=*/Vector3::UnitZ(),
      /*shape=*/LinkShape::CAPSULE,
      /*length=*/1.0}};
  std::vector<EdgeAttributes> edge_attr_stack = {EdgeAttributes{
      /*joint_pos=*/1.0,
      /*joint_rot=*/Quaternion::Identity(),
      /*scale=*/1.0}};
  std::map<std::string, NodeIndex> node_indices;

  // Pre-order traversal
  std::deque<tao::pegtl::parse_tree::node *> nodes_to_expand = {
      graph_node.get()};
  while (!nodes_to_expand.empty()) {
    auto node = nodes_to_expand.front();
    nodes_to_expand.pop_front();
    //switch (node->id) {
    //case std::type_index(typeid(dot_rules::graph)):
    //  // Expecting a name followed by statements

    for (auto &child : node->children) {
      nodes_to_expand.push_back(child.get());
    }
    if (node->has_content()) {
      std::cout << node->string() << std::endl;
    }
  }

  return graph;
}

}  // namespace robot_design
