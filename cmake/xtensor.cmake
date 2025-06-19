FetchContent_Declare(
  xtensor 
  GIT_REPOSITORY https://github.com/xtensor-stack/xtensor
  GIT_TAG        0.26.0
)

FetchContent_Declare(xtl 
  GIT_REPOSITORY https://github.com/xtensor-stack/xtl.git 
  GIT_TAG        0.8.0
)

FetchContent_MakeAvailable(xtl xtensor)
