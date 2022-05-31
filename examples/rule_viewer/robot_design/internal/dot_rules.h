#pragma once

#include <tao/pegtl.hpp>

namespace robot_design {
namespace dot_rules {

// Based on https://www.graphviz.org/doc/info/lang.html
// Unsupported features: HTML strings, ports

using namespace tao::pegtl;

// Character classes
struct ws : one<' ', '\t', '\r', '\n'> {};
struct string_first : sor<alpha, range<'\200', '\377'>, one<'_'>> {};
struct string_other : sor<string_first, digit> {};

// Keywords
struct kw_node : seq<TAO_PEGTL_ISTRING("node"), not_at<string_other>> {};
struct kw_edge : seq<TAO_PEGTL_ISTRING("edge"), not_at<string_other>> {};
struct kw_graph : seq<TAO_PEGTL_ISTRING("graph"), not_at<string_other>> {};
struct kw_digraph : seq<TAO_PEGTL_ISTRING("digraph"), not_at<string_other>> {};
struct kw_subgraph : seq<TAO_PEGTL_ISTRING("subgraph"), not_at<string_other>> {
};
struct kw_strict : seq<TAO_PEGTL_ISTRING("strict"), not_at<string_other>> {};
struct keyword
    : sor<kw_node, kw_edge, kw_graph, kw_digraph, kw_subgraph, kw_strict> {};

// ID
struct idstring : seq<not_at<keyword>, string_first, star<string_other>> {};
struct numeral
    : seq<opt<one<'-'>>, sor<seq<one<'.'>, plus<digit>>,
                             seq<plus<digit>, opt<one<'.'>, star<digit>>>>> {};
struct dqstring_content : until<at<one<'"'>>, sor<string<'\\', '"'>, any>> {};
struct dqstring : seq<one<'"'>, must<dqstring_content>, any> {
  using content = dqstring_content;
};
struct id : sor<idstring, numeral, dqstring> {};

// Operators and comments
struct edge_op : sor<string<'-', '>'>, string<'-', '-'>> {};
struct line_comment : seq<sor<two<'/'>, one<'#'>>, until<eol>> {};
struct block_comment : seq<string<'/', '*'>, until<string<'*', '/'>>> {};
struct comment : sor<line_comment, block_comment> {};

// Allowed separators
struct sep : sor<ws, comment> {};
struct seps : star<sep> {};

// https://stackoverflow.com/questions/53427551/pegtl-how-to-skip-spaces-for-the-entire-grammar
template <typename Separator, typename... Rules> struct interleaved;

template <typename Separator, typename Rule0, typename... RulesRest>
struct interleaved<Separator, Rule0, RulesRest...>
    : seq<Rule0, Separator, interleaved<Separator, RulesRest...>> {};

template <typename Separator, typename Rule0>
struct interleaved<Separator, Rule0> : seq<Rule0> {};

template <typename... Rules> using sseq = interleaved<seps, Rules...>;

// Prevents running actions if backtracking would occur
template <typename... Rules> using guarded = seq<at<Rules...>, Rules...>;

// Core grammar
struct stmt_list;
struct a_list_key : seq<id> {};
struct a_list_value : seq<id> {};
struct a_list_item
    : sseq<a_list_key, one<'='>, a_list_value, opt<one<';', ','>>> {};
struct a_list : list<a_list_item, seps> {};
struct begin_attr_list : success {};
struct attr_list
    : guarded<seq<begin_attr_list,
                  list<sseq<one<'['>, opt<a_list>, one<']'>>, seps>>> {};
struct begin_subgraph : success {};
struct subgraph_id : seq<id> {};
struct subgraph
    : guarded<sseq<begin_subgraph, opt<sseq<kw_subgraph, opt<subgraph_id>>>,
                   one<'{'>, stmt_list, one<'}'>>> {};
struct node_id : seq<id> {};
struct begin_node_stmt : success {};
struct node_attr_list : seq<attr_list> {};
struct node_stmt
    : guarded<sseq<begin_node_stmt, node_id, opt<node_attr_list>>> {};
struct edge_node_stmt : guarded<begin_node_stmt, node_id> {};
struct edge_node_arg : seq<edge_node_stmt> {};
struct edge_subgraph_arg : seq<subgraph> {};
struct edge_arg : sor<edge_node_arg, edge_subgraph_arg> {};
struct edge_rhs : list<sseq<edge_op, edge_arg>, seps> {};
struct begin_edge_stmt : success {};
struct edge_attr_list : seq<attr_list> {};
struct edge_stmt
    : guarded<sseq<begin_edge_stmt, edge_arg, edge_rhs, opt<edge_attr_list>>> {
};
struct node_def_attr_list : seq<attr_list> {};
struct edge_def_attr_list : seq<attr_list> {};
struct attr_stmt
    : sor<sseq<kw_graph, attr_list>, sseq<kw_node, node_def_attr_list>,
          sseq<kw_edge, edge_def_attr_list>> {};
struct graph_attr_key : seq<id> {};
struct graph_attr_value : seq<id> {};
struct graph_attr_stmt : sseq<graph_attr_key, one<'='>, graph_attr_value> {};
struct stmt : sor<edge_stmt, subgraph, graph_attr_stmt, attr_stmt, node_stmt> {
};
struct stmt_list : opt<list<sseq<stmt, opt<one<';'>>>, seps>> {};
struct graph_id : seq<id> {};
struct graph : sseq<opt<kw_strict>, sor<kw_graph, kw_digraph>, opt<graph_id>,
                    one<'{'>, stmt_list, one<'}'>> {};

} // namespace dot_rules
} // namespace robot_design
