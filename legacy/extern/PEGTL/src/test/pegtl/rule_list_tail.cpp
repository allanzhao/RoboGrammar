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
         verify_analyze< list_tail< eof, eof > >( __LINE__, __FILE__, false, true );
         verify_analyze< list_tail< eof, any > >( __LINE__, __FILE__, false, false );
         verify_analyze< list_tail< any, eof > >( __LINE__, __FILE__, true, false );
         verify_analyze< list_tail< any, any > >( __LINE__, __FILE__, true, false );

         verify_analyze< list_tail< eof, eof, eof > >( __LINE__, __FILE__, false, true );
         verify_analyze< list_tail< eof, eof, any > >( __LINE__, __FILE__, false, true );
         verify_analyze< list_tail< eof, any, eof > >( __LINE__, __FILE__, false, true );
         verify_analyze< list_tail< eof, any, any > >( __LINE__, __FILE__, false, false );
         verify_analyze< list_tail< any, eof, eof > >( __LINE__, __FILE__, true, true );
         verify_analyze< list_tail< any, eof, any > >( __LINE__, __FILE__, true, false );
         verify_analyze< list_tail< any, any, eof > >( __LINE__, __FILE__, true, true );
         verify_analyze< list_tail< any, any, any > >( __LINE__, __FILE__, true, false );

         verify_rule< list_tail< one< 'a' >, one< ',' > > >( __LINE__, __FILE__, "", result_type::local_failure, 0 );
         verify_rule< list_tail< one< 'a' >, one< ',' > > >( __LINE__, __FILE__, "b", result_type::local_failure, 1 );
         verify_rule< list_tail< one< 'a' >, one< ',' > > >( __LINE__, __FILE__, ",", result_type::local_failure, 1 );
         verify_rule< list_tail< one< 'a' >, one< ',' > > >( __LINE__, __FILE__, ",a", result_type::local_failure, 2 );
         verify_rule< list_tail< one< 'a' >, one< ',' > > >( __LINE__, __FILE__, "a,", result_type::success, 0 );
         verify_rule< list_tail< one< 'a' >, one< ',' > > >( __LINE__, __FILE__, "a", result_type::success, 0 );
         verify_rule< list_tail< one< 'a' >, one< ',' > > >( __LINE__, __FILE__, "a,a", result_type::success, 0 );
         verify_rule< list_tail< one< 'a' >, one< ',' > > >( __LINE__, __FILE__, "a,b", result_type::success, 1 );
         verify_rule< list_tail< one< 'a' >, one< ',' > > >( __LINE__, __FILE__, "a,a,a", result_type::success, 0 );
         verify_rule< list_tail< one< 'a' >, one< ',' > > >( __LINE__, __FILE__, "a,a,a,a", result_type::success, 0 );
         verify_rule< list_tail< one< 'a' >, one< ',' > > >( __LINE__, __FILE__, "a,a,a,b", result_type::success, 1 );
         verify_rule< list_tail< one< 'a' >, one< ',' > > >( __LINE__, __FILE__, "a,a,a,,", result_type::success, 1 );

         verify_rule< list_tail< one< 'a' >, one< ',' > > >( __LINE__, __FILE__, "a ", result_type::success, 1 );
         verify_rule< list_tail< one< 'a' >, one< ',' > > >( __LINE__, __FILE__, " a", result_type::local_failure, 2 );
         verify_rule< list_tail< one< 'a' >, one< ',' > > >( __LINE__, __FILE__, "a ,a", result_type::success, 3 );
         verify_rule< list_tail< one< 'a' >, one< ',' > > >( __LINE__, __FILE__, "a, a", result_type::success, 2 );

         verify_rule< list_must< one< 'a' >, one< ',' >, blank > >( __LINE__, __FILE__, "", result_type::local_failure, 0 );
         verify_rule< list_must< one< 'a' >, one< ',' >, blank > >( __LINE__, __FILE__, " ", result_type::local_failure, 1 );
         verify_rule< list_must< one< 'a' >, one< ',' >, blank > >( __LINE__, __FILE__, ",", result_type::local_failure, 1 );
         verify_rule< list_tail< one< 'a' >, one< ',' >, blank > >( __LINE__, __FILE__, "a ", result_type::success, 1 );
         verify_rule< list_tail< one< 'a' >, one< ',' >, blank > >( __LINE__, __FILE__, " a", result_type::local_failure, 2 );
         verify_rule< list_tail< one< 'a' >, one< ',' >, blank > >( __LINE__, __FILE__, "a ,a", result_type::success, 0 );
         verify_rule< list_tail< one< 'a' >, one< ',' >, blank > >( __LINE__, __FILE__, "a, a", result_type::success, 0 );
         verify_rule< list_tail< one< 'a' >, one< ',' >, blank > >( __LINE__, __FILE__, "a, a,", result_type::success, 0 );
         verify_rule< list_tail< one< 'a' >, one< ',' >, blank > >( __LINE__, __FILE__, "a, a ,", result_type::success, 0 );
         verify_rule< list_tail< one< 'a' >, one< ',' >, blank > >( __LINE__, __FILE__, " a , a ", result_type::local_failure, 7 );
      }

   }  // namespace TAO_PEGTL_NAMESPACE

}  // namespace tao

#include "main.hpp"
