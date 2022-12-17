// Copyright (c) 2014-2019 Dr. Colin Hirsch and Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/PEGTL/

#include "test.hpp"
#include "verify_analyze.hpp"
#include "verify_char.hpp"
#include "verify_rule.hpp"

namespace tao
{
   namespace TAO_PEGTL_NAMESPACE
   {
      void unit_test()
      {
         verify_analyze< bytes< 0 > >( __LINE__, __FILE__, false, false );

         verify_rule< bytes< 0 > >( __LINE__, __FILE__, "", result_type::success, 0 );
         verify_rule< bytes< 0 > >( __LINE__, __FILE__, "a", result_type::success, 1 );

         verify_analyze< bytes< 1 > >( __LINE__, __FILE__, true, false );

         for( char c = 0; c < 127; ++c ) {
            verify_char< bytes< 1 > >( __LINE__, __FILE__, c, result_type::success );
         }
         verify_rule< bytes< 1 > >( __LINE__, __FILE__, "", result_type::local_failure, 0 );
         verify_rule< bytes< 1 > >( __LINE__, __FILE__, "aa", result_type::success, 1 );

         verify_analyze< bytes< 2 > >( __LINE__, __FILE__, true, false );
         verify_analyze< bytes< 42 > >( __LINE__, __FILE__, true, false );

         verify_rule< bytes< 3 > >( __LINE__, __FILE__, "abcd", result_type::success, 1 );
         verify_rule< bytes< 4 > >( __LINE__, __FILE__, "abcd", result_type::success, 0 );
         verify_rule< bytes< 5 > >( __LINE__, __FILE__, "abcd", result_type::local_failure, 4 );

         verify_rule< bytes< 4 > >( __LINE__, __FILE__, "abcdefghij", result_type::success, 6 );
      }

   }  // namespace TAO_PEGTL_NAMESPACE

}  // namespace tao

#include "main.hpp"
