// Copyright (c) 2016-2019 Dr. Colin Hirsch and Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/PEGTL/

#include <clocale>
#include <cstdio>

#include "test.hpp"

namespace tao
{
   namespace TAO_PEGTL_NAMESPACE
   {
      struct file_content : seq< TAO_PEGTL_STRING( "dummy content" ), eol, discard >
      {
      };

      struct file_grammar : seq< rep_min_max< 11, 11, file_content >, eof >
      {
      };

      void unit_test()
      {
         const char* const filename = "src/test/pegtl/file_data.txt";
#if defined( _MSC_VER )
         std::FILE* stream;
         ::fopen_s( &stream, filename, "rb" );  // NOLINT
#else
         std::FILE* stream = std::fopen( filename, "rb" );  // NOLINT
#endif
         TAO_PEGTL_TEST_ASSERT( stream != nullptr );
         TAO_PEGTL_TEST_ASSERT( parse< file_grammar >( cstream_input<>( stream, 16, filename ) ) );
         std::fclose( stream );  // NOLINT(cppcoreguidelines-owning-memory)
      }

   }  // namespace TAO_PEGTL_NAMESPACE

}  // namespace tao

#include "main.hpp"
