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
         verify_analyze< if_must< any, any > >( __LINE__, __FILE__, true, false );
         verify_analyze< if_must< eof, any > >( __LINE__, __FILE__, true, false );
         verify_analyze< if_must< opt< any >, any > >( __LINE__, __FILE__, true, false );
         verify_analyze< if_must< any, opt< any > > >( __LINE__, __FILE__, true, false );
         verify_analyze< if_must< any, eof > >( __LINE__, __FILE__, true, false );
         verify_analyze< if_must< opt< any >, opt< any > > >( __LINE__, __FILE__, false, false );
         verify_analyze< if_must< eof, eof > >( __LINE__, __FILE__, false, false );

         verify_rule< if_must< one< 'a' >, one< 'b' > > >( __LINE__, __FILE__, "", result_type::local_failure, 0 );
         verify_rule< if_must< one< 'a' >, one< 'b' > > >( __LINE__, __FILE__, "a", result_type::global_failure, 0 );
         verify_rule< if_must< one< 'a' >, one< 'b' > > >( __LINE__, __FILE__, "b", result_type::local_failure, 1 );
         verify_rule< if_must< one< 'a' >, one< 'b' > > >( __LINE__, __FILE__, "ba", result_type::local_failure, 2 );
         verify_rule< if_must< one< 'a' >, one< 'b' > > >( __LINE__, __FILE__, "ab", result_type::success, 0 );
         verify_rule< if_must< one< 'a' >, one< 'b' > > >( __LINE__, __FILE__, "aba", result_type::success, 1 );
         verify_rule< if_must< one< 'a' >, one< 'b' > > >( __LINE__, __FILE__, "abb", result_type::success, 1 );
         verify_rule< if_must< one< 'a' >, one< 'b' > > >( __LINE__, __FILE__, "abab", result_type::success, 2 );
         verify_rule< if_must< one< 'a' >, one< 'b' > > >( __LINE__, __FILE__, "ac", result_type::global_failure, 1 );
         verify_rule< if_must< one< 'a' >, one< 'b' > > >( __LINE__, __FILE__, "acb", result_type::global_failure, 2 );
         verify_rule< if_must< one< 'a' >, one< 'b' >, one< 'c' > > >( __LINE__, __FILE__, "", result_type::local_failure, 0 );
         verify_rule< if_must< one< 'a' >, one< 'b' >, one< 'c' > > >( __LINE__, __FILE__, "b", result_type::local_failure, 1 );
         verify_rule< if_must< one< 'a' >, one< 'b' >, one< 'c' > > >( __LINE__, __FILE__, "bc", result_type::local_failure, 2 );
         verify_rule< if_must< one< 'a' >, one< 'b' >, one< 'c' > > >( __LINE__, __FILE__, "a", result_type::global_failure, 1 );
         verify_rule< if_must< one< 'a' >, one< 'b' >, one< 'c' > > >( __LINE__, __FILE__, "ab", result_type::global_failure, 2 );
         verify_rule< if_must< one< 'a' >, one< 'b' >, one< 'c' > > >( __LINE__, __FILE__, "ac", result_type::global_failure, 2 );
         verify_rule< if_must< one< 'a' >, one< 'b' >, one< 'c' > > >( __LINE__, __FILE__, "abb", result_type::global_failure, 3 );
         verify_rule< if_must< one< 'a' >, one< 'b' >, one< 'c' > > >( __LINE__, __FILE__, "acc", result_type::global_failure, 3 );
         verify_rule< if_must< one< 'a' >, one< 'b' >, one< 'c' > > >( __LINE__, __FILE__, "acb", result_type::global_failure, 3 );
         verify_rule< if_must< one< 'a' >, one< 'b' >, one< 'c' > > >( __LINE__, __FILE__, "abc", result_type::success, 0 );
      }

   }  // namespace TAO_PEGTL_NAMESPACE

}  // namespace tao

#include "main.hpp"
