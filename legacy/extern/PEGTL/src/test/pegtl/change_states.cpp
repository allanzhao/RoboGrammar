// Copyright (c) 2019 Dr. Colin Hirsch and Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/PEGTL/

#include "test.hpp"

namespace tao
{
   namespace TAO_PEGTL_NAMESPACE
   {
      // clang-format off
      struct A : one< 'a' > {};
      struct B : one< 'b' > {};
      struct AB : seq< A, B > {};
      // clang-format on

      template< typename >
      struct my_action
      {};

      template<>
      struct my_action< A >
      {
         static void apply0( int& c )
         {
            if( c != 0 ) {
               throw std::runtime_error( "fail1" );
            }
            c = 1;
         }
      };

      template<>
      struct my_action< B >
         : change_states< int >
      {
         static void apply0( int& v )
         {
            if( v != 0 ) {
               throw std::runtime_error( "fail6" );
            }
            v = 2;
         }

         template< typename Input >
         static void success( const Input& /*unused*/, int& v, int& c )
         {
            if( v != 2 ) {
               throw std::runtime_error( "fail3" );
            }
            if( c != 1 ) {
               throw std::runtime_error( "fail4" );
            }
            c = 3;
         }
      };

      void unit_test()
      {
         {
            memory_input<> in( "ab", "" );
            int c = 0;
            const auto result = parse< AB, my_action >( in, c );
            TAO_PEGTL_TEST_ASSERT( result );
            TAO_PEGTL_TEST_ASSERT( c == 3 );
         }
         {
            memory_input<> in( "a", "" );
            int c = 0;
            const auto result = parse< AB, my_action >( in, c );
            TAO_PEGTL_TEST_ASSERT( !result );
            TAO_PEGTL_TEST_ASSERT( c == 1 );
         }
         {
            memory_input<> in( "b", "" );
            int c = 0;
            const auto result = parse< AB, my_action >( in, c );
            TAO_PEGTL_TEST_ASSERT( !result );
            TAO_PEGTL_TEST_ASSERT( c == 0 );
         }
         {
            memory_input<> in( "ab", "" );
            int c = 5;
            const auto result = parse< disable< AB >, my_action >( in, c );
            TAO_PEGTL_TEST_ASSERT( result );
            TAO_PEGTL_TEST_ASSERT( c == 5 );
         }
      }

   }  // namespace TAO_PEGTL_NAMESPACE

}  // namespace tao

#include "main.hpp"
