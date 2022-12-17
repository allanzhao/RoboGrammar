// Copyright (c) 2014-2019 Dr. Colin Hirsch and Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/PEGTL/

#include <tao/pegtl.hpp>

#include "json_errors.hpp"

using namespace tao::TAO_PEGTL_NAMESPACE;  // NOLINT
using grammar = must< json::text, eof >;

int main( int argc, char** argv )
{
   for( int i = 1; i < argc; ++i ) {
      argv_input<> in( argv, i );
      parse< grammar, nothing, examples::errors >( in );
   }
   return 0;
}
