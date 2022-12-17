// Copyright (c) 2018-2019 Dr. Colin Hirsch and Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/PEGTL/

#include <tao/pegtl/internal/endian.hpp>

#include "test.hpp"

namespace tao
{
   namespace TAO_PEGTL_NAMESPACE
   {
      void unit_test()
      {
         TAO_PEGTL_TEST_ASSERT( internal::h_to_be( std::int8_t( 0x7a ) ) == std::int8_t( 0x7a ) );
         TAO_PEGTL_TEST_ASSERT( internal::h_to_be( std::uint8_t( 0x7a ) ) == std::uint8_t( 0x7a ) );
         TAO_PEGTL_TEST_ASSERT( internal::h_to_le( std::int8_t( 0x7a ) ) == std::int8_t( 0x7a ) );
         TAO_PEGTL_TEST_ASSERT( internal::h_to_le( std::uint8_t( 0x7a ) ) == std::uint8_t( 0x7a ) );
         TAO_PEGTL_TEST_ASSERT( internal::be_to_h( std::int8_t( 0x7a ) ) == std::int8_t( 0x7a ) );
         TAO_PEGTL_TEST_ASSERT( internal::be_to_h( std::uint8_t( 0x7a ) ) == std::uint8_t( 0x7a ) );
         TAO_PEGTL_TEST_ASSERT( internal::le_to_h( std::int8_t( 0x7a ) ) == std::int8_t( 0x7a ) );
         TAO_PEGTL_TEST_ASSERT( internal::le_to_h( std::uint8_t( 0x7a ) ) == std::uint8_t( 0x7a ) );

         const std::uint16_t a = 1;
         if( *static_cast< const std::uint8_t* >( static_cast< const void* >( &a ) ) != 1 ) {
            TAO_PEGTL_TEST_ASSERT( internal::h_to_be( std::int16_t( 0x7a39 ) ) == std::int16_t( 0x7a39 ) );
            TAO_PEGTL_TEST_ASSERT( internal::h_to_be( std::uint16_t( 0x7a39 ) ) == std::uint16_t( 0x7a39 ) );
            TAO_PEGTL_TEST_ASSERT( internal::h_to_le( std::int16_t( 0x7a39 ) ) == std::int16_t( 0x397a ) );
            TAO_PEGTL_TEST_ASSERT( internal::h_to_le( std::uint16_t( 0x7a39 ) ) == std::uint16_t( 0x397a ) );
            TAO_PEGTL_TEST_ASSERT( internal::be_to_h( std::int16_t( 0x7a39 ) ) == std::int16_t( 0x7a39 ) );
            TAO_PEGTL_TEST_ASSERT( internal::be_to_h( std::uint16_t( 0x7a39 ) ) == std::uint16_t( 0x7a39 ) );
            TAO_PEGTL_TEST_ASSERT( internal::le_to_h( std::int16_t( 0x7a39 ) ) == std::int16_t( 0x397a ) );
            TAO_PEGTL_TEST_ASSERT( internal::le_to_h( std::uint16_t( 0x7a39 ) ) == std::uint16_t( 0x397a ) );

            TAO_PEGTL_TEST_ASSERT( internal::h_to_be( std::int32_t( 0x7a391f2b ) ) == std::int32_t( 0x7a391f2b ) );
            TAO_PEGTL_TEST_ASSERT( internal::h_to_be( std::uint32_t( 0x7a391f2b ) ) == std::uint32_t( 0x7a391f2b ) );
            TAO_PEGTL_TEST_ASSERT( internal::h_to_le( std::int32_t( 0x7a391f2b ) ) == std::int32_t( 0x2b1f397a ) );
            TAO_PEGTL_TEST_ASSERT( internal::h_to_le( std::uint32_t( 0x7a391f2b ) ) == std::uint32_t( 0x2b1f397a ) );
            TAO_PEGTL_TEST_ASSERT( internal::be_to_h( std::int32_t( 0x7a391f2b ) ) == std::int32_t( 0x7a391f2b ) );
            TAO_PEGTL_TEST_ASSERT( internal::be_to_h( std::uint32_t( 0x7a391f2b ) ) == std::uint32_t( 0x7a391f2b ) );
            TAO_PEGTL_TEST_ASSERT( internal::le_to_h( std::int32_t( 0x7a391f2b ) ) == std::int32_t( 0x2b1f397a ) );
            TAO_PEGTL_TEST_ASSERT( internal::le_to_h( std::uint32_t( 0x7a391f2b ) ) == std::uint32_t( 0x2b1f397a ) );

            TAO_PEGTL_TEST_ASSERT( internal::h_to_be( std::int64_t( 0x7a391f2b33445567 ) ) == std::int64_t( 0x7a391f2b33445567 ) );
            TAO_PEGTL_TEST_ASSERT( internal::h_to_be( std::uint64_t( 0x7a391f2b33445567 ) ) == std::uint64_t( 0x7a391f2b33445567 ) );
            TAO_PEGTL_TEST_ASSERT( internal::h_to_le( std::int64_t( 0x7a391f2b33445567 ) ) == std::int64_t( 0x675544332b1f397a ) );
            TAO_PEGTL_TEST_ASSERT( internal::h_to_le( std::uint64_t( 0x7a391f2b33445567 ) ) == std::uint64_t( 0x675544332b1f397a ) );
            TAO_PEGTL_TEST_ASSERT( internal::be_to_h( std::int64_t( 0x7a391f2b33445567 ) ) == std::int64_t( 0x7a391f2b33445567 ) );
            TAO_PEGTL_TEST_ASSERT( internal::be_to_h( std::uint64_t( 0x7a391f2b33445567 ) ) == std::uint64_t( 0x7a391f2b33445567 ) );
            TAO_PEGTL_TEST_ASSERT( internal::le_to_h( std::int64_t( 0x7a391f2b33445567 ) ) == std::int64_t( 0x675544332b1f397a ) );
            TAO_PEGTL_TEST_ASSERT( internal::le_to_h( std::uint64_t( 0x7a391f2b33445567 ) ) == std::uint64_t( 0x675544332b1f397a ) );
         }
         else {
            TAO_PEGTL_TEST_ASSERT( internal::h_to_le( std::int16_t( 0x7a39 ) ) == std::int16_t( 0x7a39 ) );
            TAO_PEGTL_TEST_ASSERT( internal::h_to_le( std::uint16_t( 0x7a39 ) ) == std::uint16_t( 0x7a39 ) );
            TAO_PEGTL_TEST_ASSERT( internal::h_to_be( std::int16_t( 0x7a39 ) ) == std::int16_t( 0x397a ) );
            TAO_PEGTL_TEST_ASSERT( internal::h_to_be( std::uint16_t( 0x7a39 ) ) == std::uint16_t( 0x397a ) );
            TAO_PEGTL_TEST_ASSERT( internal::le_to_h( std::int16_t( 0x7a39 ) ) == std::int16_t( 0x7a39 ) );
            TAO_PEGTL_TEST_ASSERT( internal::le_to_h( std::uint16_t( 0x7a39 ) ) == std::uint16_t( 0x7a39 ) );
            TAO_PEGTL_TEST_ASSERT( internal::be_to_h( std::int16_t( 0x7a39 ) ) == std::int16_t( 0x397a ) );
            TAO_PEGTL_TEST_ASSERT( internal::be_to_h( std::uint16_t( 0x7a39 ) ) == std::uint16_t( 0x397a ) );

            TAO_PEGTL_TEST_ASSERT( internal::h_to_le( std::int32_t( 0x7a391f2b ) ) == std::int32_t( 0x7a391f2b ) );
            TAO_PEGTL_TEST_ASSERT( internal::h_to_le( std::uint32_t( 0x7a391f2b ) ) == std::uint32_t( 0x7a391f2b ) );
            TAO_PEGTL_TEST_ASSERT( internal::h_to_be( std::int32_t( 0x7a391f2b ) ) == std::int32_t( 0x2b1f397a ) );
            TAO_PEGTL_TEST_ASSERT( internal::h_to_be( std::uint32_t( 0x7a391f2b ) ) == std::uint32_t( 0x2b1f397a ) );
            TAO_PEGTL_TEST_ASSERT( internal::le_to_h( std::int32_t( 0x7a391f2b ) ) == std::int32_t( 0x7a391f2b ) );
            TAO_PEGTL_TEST_ASSERT( internal::le_to_h( std::uint32_t( 0x7a391f2b ) ) == std::uint32_t( 0x7a391f2b ) );
            TAO_PEGTL_TEST_ASSERT( internal::be_to_h( std::int32_t( 0x7a391f2b ) ) == std::int32_t( 0x2b1f397a ) );
            TAO_PEGTL_TEST_ASSERT( internal::be_to_h( std::uint32_t( 0x7a391f2b ) ) == std::uint32_t( 0x2b1f397a ) );

            TAO_PEGTL_TEST_ASSERT( internal::h_to_le( std::int64_t( 0x7a391f2b33445567 ) ) == std::int64_t( 0x7a391f2b33445567 ) );
            TAO_PEGTL_TEST_ASSERT( internal::h_to_le( std::uint64_t( 0x7a391f2b33445567 ) ) == std::uint64_t( 0x7a391f2b33445567 ) );
            TAO_PEGTL_TEST_ASSERT( internal::h_to_be( std::int64_t( 0x7a391f2b33445567 ) ) == std::int64_t( 0x675544332b1f397a ) );
            TAO_PEGTL_TEST_ASSERT( internal::h_to_be( std::uint64_t( 0x7a391f2b33445567 ) ) == std::uint64_t( 0x675544332b1f397a ) );
            TAO_PEGTL_TEST_ASSERT( internal::le_to_h( std::int64_t( 0x7a391f2b33445567 ) ) == std::int64_t( 0x7a391f2b33445567 ) );
            TAO_PEGTL_TEST_ASSERT( internal::le_to_h( std::uint64_t( 0x7a391f2b33445567 ) ) == std::uint64_t( 0x7a391f2b33445567 ) );
            TAO_PEGTL_TEST_ASSERT( internal::be_to_h( std::int64_t( 0x7a391f2b33445567 ) ) == std::int64_t( 0x675544332b1f397a ) );
            TAO_PEGTL_TEST_ASSERT( internal::be_to_h( std::uint64_t( 0x7a391f2b33445567 ) ) == std::uint64_t( 0x675544332b1f397a ) );
         }
      }

   }  // namespace TAO_PEGTL_NAMESPACE

}  // namespace tao

#include "main.hpp"
