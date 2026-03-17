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
    auto dB = Kokkos::View<float*>("dB", N);

    auto hA = Kokkos::create_mirror_view(dA);
    auto hB = Kokkos::create_mirror_view(dB);

    for (int i = 0; i < N; i++) {
      hA(i) = static_cast<float>(i);
      hB(i) = 0.0f;
    }

    Kokkos::deep_copy(dA, hA);
    Kokkos::deep_copy(dB, hB);

    Kokkos::parallel_for(
        "copy_blocks", Kokkos::RangePolicy<>(0, N),
        KOKKOS_LAMBDA(const int i) { dB(i) = dA(i); });
    Kokkos::fence();

    Kokkos::deep_copy(hB, dB);
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

    for (int j = 0; j < N; j++) {
      hB(j) = 0.0f;
    }
    Kokkos::deep_copy(dB, hB);

    constexpr int blockSize = 256;
    const int numBlocks = (N + blockSize - 1) / blockSize;

    Kokkos::parallel_for(
        "copy_blocks_threads", Kokkos::TeamPolicy<>(numBlocks, blockSize),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
          const int idx = team.league_rank() * team.team_size() + team.team_rank();
          if (idx < N) {
            dB(idx) = dA(idx);
          }
        });
    Kokkos::fence();

    Kokkos::deep_copy(hB, dB);
    i = 0;
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
