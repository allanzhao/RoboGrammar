// Copyright (c) 2014-2019 Dr. Colin Hirsch and Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/PEGTL/

#ifndef TAO_PEGTL_SRC_TEST_PEGTL_VERIFY_IMPL_HPP  // NOLINT
#define TAO_PEGTL_SRC_TEST_PEGTL_VERIFY_IMPL_HPP

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>

#include <tao/pegtl/apply_mode.hpp>
#include <tao/pegtl/normal.hpp>
#include <tao/pegtl/rewind_mode.hpp>

#include "result_type.hpp"
#include "test.hpp"

namespace tao
{
   namespace TAO_PEGTL_NAMESPACE
   {
      template< typename Rule, template< typename... > class Action, typename Input >
      result_type verify_impl_two( Input& in )
      {
         try {
            if( normal< Rule >::template match< apply_mode::action, rewind_mode::required, Action, normal >( in ) ) {
               return result_type::success;
            }
            return result_type::local_failure;
         }
         catch( const std::exception& ) {
            return result_type::global_failure;
         }
         catch( ... ) {
            std::cerr << "Code should be unreachable in " << __FUNCTION__ << " (" << __FILE__ << ':' << __LINE__ << ')' << std::endl;
            std::abort();
         }
      }

      template< typename Rule, template< typename... > class Action, typename Input >
      void verify_impl_one( const std::size_t line, const char* file, const std::string& data, Input& in, const result_type expected, const std::size_t remain )
      {
         const result_type received = verify_impl_two< Rule, Action >( in );

         if( ( received == expected ) && ( ( received == result_type::global_failure ) || ( in.size( 999999999 ) == remain ) ) ) {
            return;
         }
         TAO_PEGTL_TEST_FAILED( "input data [ '" << data << "' ] result received/expected [ " << received << " / " << expected << " ] remain received/expected [ " << in.size( 999999999 ) << " / " << remain << " ]" );
      }

   }  // namespace TAO_PEGTL_NAMESPACE

}  // namespace tao

#endif
