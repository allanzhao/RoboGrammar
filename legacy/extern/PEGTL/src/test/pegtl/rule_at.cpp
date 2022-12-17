// Copyright (c) 2014-2019 Dr. Colin Hirsch and Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/PEGTL/

#include "test.hpp"
#include "verify_analyze.hpp"
#include "verify_rule.hpp"

namespace tao
{
   namespace TAO_PEGTL_NAMESPACE
   {
      int at_counter = 0;

      template< typename Rule >
      struct at_action
      {
      };

      template<>
      struct at_action< any >
      {
         template< typename Input >
         static void apply( const Input& /*unused*/ )
         {
            ++at_counter;
         }
      };

      void unit_test()
      {
         TAO_PEGTL_TEST_ASSERT( at_counter == 0 );

         verify_analyze< at< eof > >( __LINE__, __FILE__, false, false );
         verify_analyze< at< any > >( __LINE__, __FILE__, false, false );

         verify_rule< at< eof > >( __LINE__, __FILE__, "", result_type::success, 0 );
         verify_rule< at< eof > >( __LINE__, __FILE__, "a", result_type::local_failure, 1 );
         verify_rule< at< any > >( __LINE__, __FILE__, "", result_type::local_failure, 0 );
         verify_rule< at< any > >( __LINE__, __FILE__, "a", result_type::success, 1 );
         verify_rule< at< any > >( __LINE__, __FILE__, "aa", result_type::success, 2 );
         verify_rule< at< any > >( __LINE__, __FILE__, "aaaa", result_type::success, 4 );
         verify_rule< must< at< alpha > > >( __LINE__, __FILE__, "1", result_type::global_failure, 1 );
         verify_rule< must< at< alpha, alpha > > >( __LINE__, __FILE__, "a1a", result_type::global_failure, 3 );
         {
            memory_input<> in( "f", 1, __FILE__ );
            parse< any, at_action >( in );
            TAO_PEGTL_TEST_ASSERT( at_counter == 1 );
         }
         {
            memory_input<> in( "f", 1, __FILE__ );
            parse< at< any >, at_action >( in );
            TAO_PEGTL_TEST_ASSERT( at_counter == 1 );
         }
      }

   }  // namespace TAO_PEGTL_NAMESPACE

}  // namespace tao

#include "main.hpp"
