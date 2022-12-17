# Migration Guide

## Version 2.8.0

* All enumerators were changed from uppercase to lowercase.
* The enumerator `tracking_mode::IMMEDIATE` was renamed to `tracking_mode::eager`.
* The input-classes' `peek_byte()` member function was renamed to `peek_uint8()`.

Compatibility enumerators and functions are provided, they will be removed in version 3.0.0.

## Version 2.4.0

* The common prefix for all macros was changed from `TAOCPP_PEGTL_` to `TAO_PEGTL_`.

Compatibility macros are provided, they will be removed in version 3.0.0.

## Version 2.0.0

Some of the [changes](Changelog.md#200) for version 2.0.0 require modifications to any source using the PEGTL.
The good news is that the semantics of all parsing rules and grammars is the same as for versions 1.x.
Existing grammars will continue to work as before once the following list of mostly naming and namespace related changes is taken into account.

* Everything in the PEGTL is now in namespace `tao::pegtl`.
* The file name extensions were changed from `.hh` to `.hpp`.
* The main include file is now `<tao/pegtl.hpp>`, all other include files are `<tao/pegtl/*.hpp>`.
* The distribution of tasks between [the parse functions and input classes](Inputs-and-Parsing.md) was changed.
* The string macros have been renamed from `pegtl_(i)string_t` to `TAOCPP_PEGTL_(I)STRING`.
* The `begin()` member function of the input-classes is now called `current()`.
* The first argument to actions' `apply()` is now of type `tao::pegtl::internal::action_input< ... >`.

For flexibility and future compatibility it is recommended to "template over" the first argument to `apply()` as shown in [Actions and States](Actions-and-States.md#actions).

The `pegtl` part of the namespace can be controlled with the `TAOCPP_PEGTL_NAMESPACE` macro which is set in `include/tao/pegtl/config.hpp` if not defined previously.

Most of the other changes for version 2.0.0 extend or optimise the PEGTL without breaking compatibility.
Applications integrated more tightly with the PEGTL might require additional changes.

Please contact the authors at `taocpp(at)icemx.net` for any further questions when updating the PEGTL.

## Version 1.0.0

There were [many important changes](Changelog.md#100) leading up to version 1.0.0.
Please contact the authors at `taocpp(at)icemx.net` for any further questions when updating the PEGTL.

Copyright (c) 2017-2019 Dr. Colin Hirsch and Daniel Frey
