# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.13.1/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.13.1/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/a.e./Dropbox/exact

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/a.e./Dropbox/exact/build

# Include any dependencies generated for this target.
include rnn_tests/CMakeFiles/test_rnn.dir/depend.make

# Include the progress variables for this target.
include rnn_tests/CMakeFiles/test_rnn.dir/progress.make

# Include the compile flags for this target's objects.
include rnn_tests/CMakeFiles/test_rnn.dir/flags.make

rnn_tests/CMakeFiles/test_rnn.dir/test_rnn.cxx.o: rnn_tests/CMakeFiles/test_rnn.dir/flags.make
rnn_tests/CMakeFiles/test_rnn.dir/test_rnn.cxx.o: ../rnn_tests/test_rnn.cxx
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/a.e./Dropbox/exact/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object rnn_tests/CMakeFiles/test_rnn.dir/test_rnn.cxx.o"
	cd /Users/a.e./Dropbox/exact/build/rnn_tests && /Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_rnn.dir/test_rnn.cxx.o -c /Users/a.e./Dropbox/exact/rnn_tests/test_rnn.cxx

rnn_tests/CMakeFiles/test_rnn.dir/test_rnn.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_rnn.dir/test_rnn.cxx.i"
	cd /Users/a.e./Dropbox/exact/build/rnn_tests && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/a.e./Dropbox/exact/rnn_tests/test_rnn.cxx > CMakeFiles/test_rnn.dir/test_rnn.cxx.i

rnn_tests/CMakeFiles/test_rnn.dir/test_rnn.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_rnn.dir/test_rnn.cxx.s"
	cd /Users/a.e./Dropbox/exact/build/rnn_tests && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/a.e./Dropbox/exact/rnn_tests/test_rnn.cxx -o CMakeFiles/test_rnn.dir/test_rnn.cxx.s

# Object files for target test_rnn
test_rnn_OBJECTS = \
"CMakeFiles/test_rnn.dir/test_rnn.cxx.o"

# External object files for target test_rnn
test_rnn_EXTERNAL_OBJECTS =

rnn_tests/test_rnn: rnn_tests/CMakeFiles/test_rnn.dir/test_rnn.cxx.o
rnn_tests/test_rnn: rnn_tests/CMakeFiles/test_rnn.dir/build.make
rnn_tests/test_rnn: rnn/libexalt_strategy.a
rnn_tests/test_rnn: common/libexact_common.a
rnn_tests/test_rnn: time_series/libexact_time_series.a
rnn_tests/test_rnn: /usr/local/lib/libmysqlclient.dylib
rnn_tests/test_rnn: rnn_tests/CMakeFiles/test_rnn.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/a.e./Dropbox/exact/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test_rnn"
	cd /Users/a.e./Dropbox/exact/build/rnn_tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_rnn.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
rnn_tests/CMakeFiles/test_rnn.dir/build: rnn_tests/test_rnn

.PHONY : rnn_tests/CMakeFiles/test_rnn.dir/build

rnn_tests/CMakeFiles/test_rnn.dir/clean:
	cd /Users/a.e./Dropbox/exact/build/rnn_tests && $(CMAKE_COMMAND) -P CMakeFiles/test_rnn.dir/cmake_clean.cmake
.PHONY : rnn_tests/CMakeFiles/test_rnn.dir/clean

rnn_tests/CMakeFiles/test_rnn.dir/depend:
	cd /Users/a.e./Dropbox/exact/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/a.e./Dropbox/exact /Users/a.e./Dropbox/exact/rnn_tests /Users/a.e./Dropbox/exact/build /Users/a.e./Dropbox/exact/build/rnn_tests /Users/a.e./Dropbox/exact/build/rnn_tests/CMakeFiles/test_rnn.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : rnn_tests/CMakeFiles/test_rnn.dir/depend
