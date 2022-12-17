#!/usr/bin/env python
# -*- coding: utf-8 -*-
from conans import ConanFile, CMake

class PEGTLConan(ConanFile):
    name = "pegtl"
    description = "C++ header-only parser combinator library for creating PEG parsers"
    homepage = "https://github.com/taocpp/PEGTL"
    topics = ("conan", "taocpp", "pegtl", "peg", "grammar", "parsing")
    url = homepage
    license = "MIT"
    author = "taocpp@icemx.net"
    exports = "LICENSE"
    exports_sources = "cmake/*", "include/*", "CMakeLists.txt"
    settings = "build_type", "compiler", "os", "arch"
    generators = "cmake"
    no_copy_source = True

    def build(self):
        pass

    def package(self):
        cmake = CMake(self)
        cmake.definitions["PEGTL_BUILD_TESTS"] = "OFF"
        cmake.definitions["PEGTL_BUILD_EXAMPLES"] = "OFF"
        cmake.definitions["PEGTL_INSTALL_DOC_DIR"] = "licenses"
        cmake.configure()
        cmake.install()

    def package_id(self):
        self.info.header_only()
