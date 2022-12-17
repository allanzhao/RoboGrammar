// Copyright (c) 2017-2019 Dr. Colin Hirsch and Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/PEGTL/

#include "test.hpp"
#include "verify_seqs.hpp"

namespace tao
{
   namespace TAO_PEGTL_NAMESPACE
   {
      namespace test1
      {
         struct action_a
         {
            template< typename Input >
            static void apply( const Input& in, std::string& r, std::string& s )
            {
               TAO_PEGTL_TEST_ASSERT( r.empty() );
               TAO_PEGTL_TEST_ASSERT( s.empty() );
               r += in.string();
            }
         };

         struct action_b
         {
            template< typename Input >
            static void apply( const Input& in, std::string& r, std::string& s )
            {
               TAO_PEGTL_TEST_ASSERT( s.empty() );
               s += in.string();
               s += "*";
               s += r;
            }
         };

         struct action2_a
         {
            template< typename Input >
            static void apply( const Input& in, bool& state_b )
            {
               TAO_PEGTL_TEST_ASSERT( in.string() == "foo" );
               TAO_PEGTL_TEST_ASSERT( !state_b );
            }
         };

         struct action2_b
         {
            template< typename Input >
            static bool apply( const Input& in, bool& state_b )
            {
               TAO_PEGTL_TEST_ASSERT( in.string() == "foo" );
               TAO_PEGTL_TEST_ASSERT( !state_b );
               state_b = true;
               return false;
            }
         };

         struct action2_c
         {
            template< typename Input >
            static void apply( const Input& /*unused*/, bool& /*unused*/ )
            {
               TAO_PEGTL_TEST_ASSERT( false );
            }
         };

         template< typename Rule >
         struct action
         {
         };

         int flag = 0;

         template<>
         struct action< one< '-' > >
         {
            static void apply0( std::string& /*unused*/, std::string& /*unused*/ )
            {
               ++flag;
            }
         };

      }  // namespace test1

      template< typename... Rules >
      using if_apply_seq = if_apply< seq< Rules... > >;

      template< typename... Rules >
      using if_apply_disable = if_apply< disable< Rules... > >;

      void unit_test()
      {
         std::string state_r;
         std::string state_s;
         TAO_PEGTL_TEST_ASSERT( test1::flag == 0 );
         memory_input<> in1( "-", __FUNCTION__ );
         parse< must< if_apply< one< '-' >, test1::action_a, test1::action_b > >, test1::action >( in1, state_r, state_s );
         TAO_PEGTL_TEST_ASSERT( test1::flag == 1 );
         TAO_PEGTL_TEST_ASSERT( state_r == "-" );
         TAO_PEGTL_TEST_ASSERT( state_s == "-*-" );
         memory_input<> in2( "-", __FUNCTION__ );
         parse< must< disable< if_apply< one< '-' >, test1::action_a, test1::action_b > > >, test1::action >( in2, state_r, state_s );
         TAO_PEGTL_TEST_ASSERT( test1::flag == 1 );
         TAO_PEGTL_TEST_ASSERT( state_r == "-" );
         TAO_PEGTL_TEST_ASSERT( state_s == "-*-" );

         {
            bool state_b = false;
            const bool result = parse< if_apply< plus< alpha >, test1::action2_a, test1::action2_b, test1::action2_c > >( memory_input<>( "foo bar", __FUNCTION__ ), state_b );
            TAO_PEGTL_TEST_ASSERT( !result );
            TAO_PEGTL_TEST_ASSERT( state_b );
         }

         {
            bool state_b = false;
            const bool result = parse< if_apply< plus< alpha >, test1::action2_a, test1::action2_b, test1::action2_c > >( memory_input<>( "", __FUNCTION__ ), state_b );
            TAO_PEGTL_TEST_ASSERT( !result );
            TAO_PEGTL_TEST_ASSERT( !state_b );
         }

         verify_seqs< if_apply_seq >();
         verify_seqs< if_apply_disable >();
      }

   }  // namespace TAO_PEGTL_NAMESPACE

}  // namespace tao

#include "main.hpp"
