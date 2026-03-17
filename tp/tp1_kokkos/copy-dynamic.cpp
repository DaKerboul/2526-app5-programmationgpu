#include <Kokkos_Core.hpp>
#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " N" << std::endl;
    return 0;
  }

  const int N = std::atoi(argv[1]);

  Kokkos::initialize(argc, argv);
  {
    auto dA = Kokkos::View<float*>("dA", N);
    auto hA = Kokkos::create_mirror_view(dA);
    auto hB = Kokkos::create_mirror_view(dA);

    for (int i = 0; i < N; i++) {
      hA(i) = static_cast<float>(i);
      hB(i) = 0.0f;
    }

    Kokkos::deep_copy(dA, hA);
    Kokkos::deep_copy(hB, dA);

    int i = 0;
    for (; i < N; i++) {
      if (hA(i) != hB(i)) {
        break;
      }
    }

    if (i < N) {
      std::cout << "La copie est incorrecte!" << std::endl;
    } else {
      std::cout << "La copie est correcte!" << std::endl;
    }
  }
  Kokkos::finalize();
  return 0;
}
