# Install script for directory: /Users/a.e./Dropbox/newAntColony

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/Users/a.e./Dropbox/newAntColony/build/common/cmake_install.cmake")
  include("/Users/a.e./Dropbox/newAntColony/build/image_tools/cmake_install.cmake")
  include("/Users/a.e./Dropbox/newAntColony/build/time_series/cmake_install.cmake")
  include("/Users/a.e./Dropbox/newAntColony/build/cnn/cmake_install.cmake")
  include("/Users/a.e./Dropbox/newAntColony/build/rnn/cmake_install.cmake")
  include("/Users/a.e./Dropbox/newAntColony/build/rnn_tests/cmake_install.cmake")
  include("/Users/a.e./Dropbox/newAntColony/build/rnn_examples/cmake_install.cmake")
  include("/Users/a.e./Dropbox/newAntColony/build/cnn_tests/cmake_install.cmake")
  include("/Users/a.e./Dropbox/newAntColony/build/cnn_examples/cmake_install.cmake")
  include("/Users/a.e./Dropbox/newAntColony/build/multithreaded/cmake_install.cmake")
  include("/Users/a.e./Dropbox/newAntColony/build/mpi/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/Users/a.e./Dropbox/newAntColony/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
