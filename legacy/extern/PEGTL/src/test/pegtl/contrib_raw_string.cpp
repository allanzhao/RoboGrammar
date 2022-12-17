// Copyright (c) 2016-2019 Dr. Colin Hirsch and Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/PEGTL/

#include "test.hpp"
#include "verify_fail.hpp"

#include <tao/pegtl/contrib/raw_string.hpp>

namespace tao
{
   namespace TAO_PEGTL_NAMESPACE
   {
      std::string content;  // NOLINT

      using rstring = raw_string< '[', '=', ']' >;
      using qstring = raw_string< '[', '=', ']', alpha, digit >;

      template< typename Rule >
      struct raction
      {
      };

      template<>
      struct raction< rstring::content >
      {
         template< typename Input, typename... States >
         static void apply( const Input& in, const States&... /*unused*/ )
         {
            content.assign( in.begin(), in.end() );
         }
      };

      template< typename Rule >
      struct qaction
      {
      };

      template<>
      struct qaction< qstring::content >
      {
         template< typename Input, typename... States >
         static void apply( const Input& in, const States&... /*unused*/ )
         {
            content.assign( in.begin(), in.end() );
         }
      };

      struct rgrammar
         : must< rstring, eof >
      {
      };

      struct qgrammar
         : must< qstring, eof >
      {
      };

      template< typename Rule, template< typename > class Action, unsigned M, unsigned N >
      void verify_data( const std::size_t line, const char* file, const char ( &m )[ M ], const char ( &n )[ N ] )  // NOLINT
      {
         content.clear();
         memory_input<> in( m, m + M - 1, file, 0, line, 0 );
         const auto r = parse< Rule, Action >( in );
         if( ( !r ) || ( content != std::string( n, N - 1 ) ) ) {
            TAO_PEGTL_TEST_FAILED( "input data [ '" << m << "' ] expected success with [ '" << n << "' ] but got [ '" << content << "' ] result [ " << r << " ]" );
         }
         content.clear();
         memory_input< tracking_mode::lazy > in2( m, m + M - 1, file, 0, line, 0 );
         const auto r2 = parse< Rule, Action >( in2 );
         if( ( !r2 ) || ( content != std::string( n, N - 1 ) ) ) {
            TAO_PEGTL_TEST_FAILED( "input data [ '" << m << "' ] with tracking_mode::lazy expected success with [ '" << n << "' ] but got [ '" << content << "' ] result [ " << r2 << " ]" );
         }
      }

      void unit_test()
      {
         verify_data< rgrammar, raction >( __LINE__, __FILE__, "[[]]", "" );
         verify_data< rgrammar, raction >( __LINE__, __FILE__, "[[foo]]", "foo" );
         verify_data< rgrammar, raction >( __LINE__, __FILE__, "[===[foo]===]", "foo" );
         verify_data< rgrammar, raction >( __LINE__, __FILE__, "[===[\nfoo]===]", "foo" );
         verify_data< rgrammar, raction >( __LINE__, __FILE__, "[===[\r\nfoo]===]", "foo" );
         verify_data< rgrammar, raction >( __LINE__, __FILE__, "[===[\0\0\0]===]", "\0\0\0" );

         verify_data< qgrammar, qaction >( __LINE__, __FILE__, "[[]]", "" );
         verify_data< qgrammar, qaction >( __LINE__, __FILE__, "[[a1]]", "a1" );
         verify_data< qgrammar, qaction >( __LINE__, __FILE__, "[===[a1]===]", "a1" );
         verify_data< qgrammar, qaction >( __LINE__, __FILE__, "[===[\na1]===]", "a1" );
         verify_data< qgrammar, qaction >( __LINE__, __FILE__, "[===[\r\na1]===]", "a1" );
         verify_data< qgrammar, qaction >( __LINE__, __FILE__, "[===[a0a1a2a3]===]", "a0a1a2a3" );

         verify_fail< rgrammar >( __LINE__, __FILE__, "" );
         verify_fail< rgrammar >( __LINE__, __FILE__, "[" );
         verify_fail< rgrammar >( __LINE__, __FILE__, "[=" );
         verify_fail< rgrammar >( __LINE__, __FILE__, "[=[" );
         verify_fail< rgrammar >( __LINE__, __FILE__, "[=[]=" );
         verify_fail< rgrammar >( __LINE__, __FILE__, "[=[]]" );
         verify_fail< rgrammar >( __LINE__, __FILE__, "[]" );
         verify_fail< rgrammar >( __LINE__, __FILE__, "[[]] " );
         verify_fail< rgrammar >( __LINE__, __FILE__, " [[]]" );
         verify_fail< rgrammar >( __LINE__, __FILE__, "[=[]-]" );
         verify_fail< rgrammar >( __LINE__, __FILE__, "[-[]=]" );
         verify_fail< rgrammar >( __LINE__, __FILE__, "[-[]-]" );
         verify_fail< rgrammar >( __LINE__, __FILE__, "[===[]====]" );
         verify_fail< rgrammar >( __LINE__, __FILE__, "[====[]===]" );

         verify_fail< qgrammar >( __LINE__, __FILE__, "" );
         verify_fail< qgrammar >( __LINE__, __FILE__, "[" );
         verify_fail< qgrammar >( __LINE__, __FILE__, "[=" );
         verify_fail< qgrammar >( __LINE__, __FILE__, "[=[" );
         verify_fail< qgrammar >( __LINE__, __FILE__, "[=[]=" );
         verify_fail< qgrammar >( __LINE__, __FILE__, "[=[]]" );
         verify_fail< qgrammar >( __LINE__, __FILE__, "[]" );
         verify_fail< qgrammar >( __LINE__, __FILE__, "[[]] " );
         verify_fail< qgrammar >( __LINE__, __FILE__, " [[]]" );
         verify_fail< qgrammar >( __LINE__, __FILE__, "[=[]-]" );
         verify_fail< qgrammar >( __LINE__, __FILE__, "[-[]=]" );
         verify_fail< qgrammar >( __LINE__, __FILE__, "[-[]-]" );
         verify_fail< qgrammar >( __LINE__, __FILE__, "[===[]====]" );
         verify_fail< qgrammar >( __LINE__, __FILE__, "[====[]===]" );

         verify_fail< qgrammar >( __LINE__, __FILE__, "[=[-]=]" );
         verify_fail< qgrammar >( __LINE__, __FILE__, "[=[1]=]" );
         verify_fail< qgrammar >( __LINE__, __FILE__, "[=[a]=]" );
         verify_fail< qgrammar >( __LINE__, __FILE__, "[=[a+]=]" );
         verify_fail< qgrammar >( __LINE__, __FILE__, "[=[aa]=]" );
         verify_fail< qgrammar >( __LINE__, __FILE__, "[=[11]=]" );
         verify_fail< qgrammar >( __LINE__, __FILE__, "[=[a1a]=]" );
         verify_fail< qgrammar >( __LINE__, __FILE__, "[=[a1aa]=]" );
      }

   }  // namespace TAO_PEGTL_NAMESPACE

}  // namespace tao

#include "main.hpp"
