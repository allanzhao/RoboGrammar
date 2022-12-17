// Copyright (c) 2014-2019 Dr. Colin Hirsch and Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/PEGTL/

#include "test.hpp"
#include "verify_analyze.hpp"
#include "verify_rule.hpp"

namespace tao
{
   namespace TAO_PEGTL_NAMESPACE
   {
      void unit_test()
      {
         verify_analyze< plus< eof > >( __LINE__, __FILE__, false, true );
         verify_analyze< plus< any > >( __LINE__, __FILE__, true, false );
         verify_analyze< plus< eof, eof, eof > >( __LINE__, __FILE__, false, true );
         verify_analyze< plus< any, eof, any > >( __LINE__, __FILE__, true, false );

         verify_rule< plus< one< 'a' > > >( __LINE__, __FILE__, "", result_type::local_failure, 0 );
         verify_rule< plus< one< 'a' > > >( __LINE__, __FILE__, "a", result_type::success, 0 );
         verify_rule< plus< one< 'a' > > >( __LINE__, __FILE__, "aa", result_type::success, 0 );
         verify_rule< plus< one< 'a' > > >( __LINE__, __FILE__, "aaa", result_type::success, 0 );
         verify_rule< plus< one< 'a' > > >( __LINE__, __FILE__, "b", result_type::local_failure, 1 );
         verify_rule< plus< one< 'a' > > >( __LINE__, __FILE__, "ab", result_type::success, 1 );
         verify_rule< plus< one< 'a' > > >( __LINE__, __FILE__, "aab", result_type::success, 1 );
         verify_rule< plus< one< 'a' > > >( __LINE__, __FILE__, "aaab", result_type::success, 1 );

         verify_rule< plus< one< 'a' >, one< 'b' > > >( __LINE__, __FILE__, "", result_type::local_failure, 0 );
         verify_rule< plus< one< 'a' >, one< 'b' > > >( __LINE__, __FILE__, "a", result_type::local_failure, 1 );
         verify_rule< plus< one< 'a' >, one< 'b' > > >( __LINE__, __FILE__, "b", result_type::local_failure, 1 );
         verify_rule< plus< one< 'a' >, one< 'b' > > >( __LINE__, __FILE__, "ab", result_type::success, 0 );
         verify_rule< plus< one< 'a' >, one< 'b' > > >( __LINE__, __FILE__, "ac", result_type::local_failure, 2 );
         verify_rule< plus< one< 'a' >, one< 'b' > > >( __LINE__, __FILE__, "aa", result_type::local_failure, 2 );
         verify_rule< plus< one< 'a' >, one< 'b' > > >( __LINE__, __FILE__, "aba", result_type::success, 1 );
         verify_rule< plus< one< 'a' >, one< 'b' > > >( __LINE__, __FILE__, "abb", result_type::success, 1 );
         verify_rule< plus< one< 'a' >, one< 'b' > > >( __LINE__, __FILE__, "abc", result_type::success, 1 );
         verify_rule< plus< one< 'a' >, one< 'b' > > >( __LINE__, __FILE__, "abab", result_type::success, 0 );
         verify_rule< plus< one< 'a' >, one< 'b' > > >( __LINE__, __FILE__, "ababa", result_type::success, 1 );
         verify_rule< plus< one< 'a' >, one< 'b' > > >( __LINE__, __FILE__, "ababb", result_type::success, 1 );
      }

   }  // namespace TAO_PEGTL_NAMESPACE

}  // namespace tao

#include "main.hpp"
