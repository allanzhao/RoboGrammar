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
         verify_analyze< pad_opt< eof, eof > >( __LINE__, __FILE__, false, true );
         verify_analyze< pad_opt< eof, any > >( __LINE__, __FILE__, false, false );
         verify_analyze< pad_opt< any, eof > >( __LINE__, __FILE__, false, true );
         verify_analyze< pad_opt< any, any > >( __LINE__, __FILE__, false, false );

         verify_rule< pad_opt< one< 'a' >, space > >( __LINE__, __FILE__, "", result_type::success, 0 );
         verify_rule< pad_opt< one< 'a' >, space > >( __LINE__, __FILE__, " ", result_type::success, 0 );
         verify_rule< pad_opt< one< 'a' >, space > >( __LINE__, __FILE__, "  ", result_type::success, 0 );
         verify_rule< pad_opt< one< 'a' >, space > >( __LINE__, __FILE__, "b", result_type::success, 1 );
         verify_rule< pad_opt< one< 'a' >, space > >( __LINE__, __FILE__, "ba", result_type::success, 2 );
         verify_rule< pad_opt< one< 'a' >, space > >( __LINE__, __FILE__, "a", result_type::success, 0 );
         verify_rule< pad_opt< one< 'a' >, space > >( __LINE__, __FILE__, " a", result_type::success, 0 );
         verify_rule< pad_opt< one< 'a' >, space > >( __LINE__, __FILE__, "a ", result_type::success, 0 );
         verify_rule< pad_opt< one< 'a' >, space > >( __LINE__, __FILE__, "  a", result_type::success, 0 );
         verify_rule< pad_opt< one< 'a' >, space > >( __LINE__, __FILE__, "  b", result_type::success, 1 );
         verify_rule< pad_opt< one< 'a' >, space > >( __LINE__, __FILE__, "a  ", result_type::success, 0 );
         verify_rule< pad_opt< one< 'a' >, space > >( __LINE__, __FILE__, "  a  ", result_type::success, 0 );
         verify_rule< pad_opt< one< 'a' >, space > >( __LINE__, __FILE__, "   a   ", result_type::success, 0 );
         verify_rule< pad_opt< one< 'a' >, space > >( __LINE__, __FILE__, "aa", result_type::success, 1 );
         verify_rule< pad_opt< one< 'a' >, space > >( __LINE__, __FILE__, "a a", result_type::success, 1 );
         verify_rule< pad_opt< one< 'a' >, space > >( __LINE__, __FILE__, "  a  a ", result_type::success, 2 );
      }

   }  // namespace TAO_PEGTL_NAMESPACE

}  // namespace tao

#include "main.hpp"
