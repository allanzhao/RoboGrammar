// Copyright (c) 2014-2019 Dr. Colin Hirsch and Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/PEGTL/

#ifndef TAO_PEGTL_SRC_TEST_PEGTL_VERIFY_ANALYZE_HPP  // NOLINT
#define TAO_PEGTL_SRC_TEST_PEGTL_VERIFY_ANALYZE_HPP

#include <tao/pegtl/analyze.hpp>

#include "test.hpp"

namespace tao
{
   namespace TAO_PEGTL_NAMESPACE
   {
      template< typename Rule >
      void verify_analyze( const unsigned line, const char* file, const bool expect_consume, const bool expect_problems )
      {
         analysis::analyze_cycles< Rule > a( false );

         const bool has_problems = ( a.problems() != 0 );
         const bool does_consume = a.template consumes< Rule >();

         if( has_problems != expect_problems ) {
            TAO_PEGTL_TEST_FAILED( "analyze -- problems received/expected [ " << has_problems << " / " << expect_problems << " ]" );
         }
         if( does_consume != expect_consume ) {
            TAO_PEGTL_TEST_FAILED( "analyze -- consumes received/expected [ " << does_consume << " / " << expect_consume << " ]" );
         }
      }

   }  // namespace TAO_PEGTL_NAMESPACE

}  // namespace tao

#endif
