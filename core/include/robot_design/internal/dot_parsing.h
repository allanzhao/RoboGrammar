#pragma once

#include <iostream>  // TODO
#include <map>
#include <robot_design/internal/dot_rules.h>
#include <string>
#include <tao/pegtl.hpp>
#include <tao/pegtl/contrib/parse_tree.hpp>

namespace robot_design {
namespace dot_parsing {

template <typename Rule>
using dot_selector = tao::pegtl::parse_tree::selector<Rule,
    tao::pegtl::parse_tree::store_content::on<
        dot_rules::id>,
    tao::pegtl::parse_tree::remove_content::on<
        dot_rules::a_list_item,
        dot_rules::attr_stmt,
        dot_rules::edge_stmt,
        dot_rules::graph,
        dot_rules::graph_attr_stmt,
        dot_rules::node_stmt,
        dot_rules::subgraph>>;

struct AttrContext {
  std::map<std::string, std::string> attr_list_items_;
};

}  // namespace dot_parsing
}  // namespace robot_design
