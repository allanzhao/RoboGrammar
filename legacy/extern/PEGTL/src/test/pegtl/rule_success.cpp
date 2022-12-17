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
         verify_analyze< success >( __LINE__, __FILE__, false, false );

         verify_rule< success >( __LINE__, __FILE__, "", result_type::success, 0 );

         for( char i = 1; i < 127; ++i ) {
            char t[] = { i, 0 };  // NOLINT
            verify_rule< success >( __LINE__, __FILE__, std::string( t ), result_type::success, 1 );
         }
      }

   }  // namespace TAO_PEGTL_NAMESPACE

}  // namespace tao

#include "main.hpp"
