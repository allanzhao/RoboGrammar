#pragma once

#include <tao/pegtl.hpp>

namespace robot_design {
namespace dot_rules {

// Based on https://www.graphviz.org/doc/info/lang.html
// Changes:
// 1) Rules modified to reduce backtracking
// 2) No support for HTML strings
// 3) No special handling for compass points (simply parsed as IDs)

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
struct kw_subgraph : seq<TAO_PEGTL_ISTRING("subgraph"), not_at<string_other>> {};
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
template <typename Separator, typename... Rules>
struct interleaved;

template <typename Separator, typename Rule0, typename... RulesRest>
struct interleaved<Separator, Rule0, RulesRest...>
    : seq<Rule0, Separator, interleaved<Separator, RulesRest...>> {};

template <typename Separator, typename Rule0>
struct interleaved<Separator, Rule0>
    : seq<Rule0> {};

template <typename... Rules>
using sseq = interleaved<seps, Rules...>;

// Core grammar
struct stmt_list;
struct attr_list;
struct subgraph
    : sseq<opt<sseq<kw_subgraph, opt<id>>>, one<'{'>, stmt_list, one<'}'>> {};
struct port
    : sseq<one<':'>, id, opt<sseq<one<':'>, id>>> {};
struct node_id : sseq<id, opt<port>> {};
struct node_stmt : sseq<node_id, opt<attr_list>> {};
struct edge_rhs : list<sseq<edge_op, sor<node_id, subgraph>>, seps> {};
struct edge_stmt : sseq<sor<node_id, subgraph>, edge_rhs, opt<attr_list>> {};
struct a_list_item
    : sseq<id, one<'='>, id, opt<one<';', ','>>> {};
struct a_list : list<a_list_item, seps> {};
struct attr_list : list<sseq<one<'['>, opt<a_list>, one<']'>>, seps> {};
struct attr_stmt : sseq<sor<kw_graph, kw_node, kw_edge>, attr_list> {};
struct graph_attr_stmt : sseq<id, one<'='>, id> {};
struct stmt
    : sor<subgraph, graph_attr_stmt, attr_stmt, edge_stmt, node_stmt> {};
struct stmt_list : opt<list<sseq<stmt, opt<one<';'>>>, seps>> {};
struct graph
    : sseq<opt<kw_strict>, sor<kw_graph, kw_digraph>, opt<id>, one<'{'>,
           stmt_list, one<'}'>> {};

}  // namespace dot_rules
}  // namespace robot_design
