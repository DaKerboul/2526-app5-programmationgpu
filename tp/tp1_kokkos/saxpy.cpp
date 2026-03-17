#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

template <class ViewX, class ViewY, class ViewRes>
void verifySaxpy(float a, const ViewX& x, const ViewY& y, const ViewRes& res,
                 int N) {
  int i = 0;
  for (; i < N; i++) {
    const float expected = a * x(i) + y(i);
    const float denom = std::abs(expected) > 1e-6f ? std::abs(expected) : 1e-6f;
    if (std::abs(res(i) - expected) / denom > 1e-6f) {
      std::cout << res(i) << " " << expected << std::endl;
      break;
    }
  }

  if (i == N) {
    std::cout << "saxpy on Kokkos is correct." << std::endl;
  } else {
    std::cout << "saxpy on Kokkos is incorrect on element " << i << "." << std::endl;
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "Utilisation: " << argv[0] << " N" << std::endl;
    return 0;
  }

  const int N = std::atoi(argv[1]);
  constexpr float a = 2.0f;

  Kokkos::initialize(argc, argv);
  {
    auto dx = Kokkos::View<float*>("dx", N);
    auto dy = Kokkos::View<float*>("dy", N);

    auto hx = Kokkos::create_mirror_view(dx);
    auto hy = Kokkos::create_mirror_view(dy);
    auto hres = Kokkos::create_mirror_view(dy);

    for (int i = 0; i < N; i++) {
      hx(i) = static_cast<float>(i);
      hy(i) = 1.0f;
    }

    Kokkos::deep_copy(dx, hx);
    Kokkos::deep_copy(dy, hy);

    Kokkos::parallel_for(
        "saxpy_blocks", Kokkos::RangePolicy<>(0, N),
        KOKKOS_LAMBDA(const int i) { dy(i) = a * dx(i) + dy(i); });
    Kokkos::fence();
    Kokkos::deep_copy(hres, dy);
    verifySaxpy(a, hx, hy, hres, N);

    Kokkos::deep_copy(dy, hy);
    constexpr int blockSize = 256;
    const int numBlocks = (N + blockSize - 1) / blockSize;

    Kokkos::parallel_for(
        "saxpy_blocks_threads", Kokkos::TeamPolicy<>(numBlocks, blockSize),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
          const int i = team.league_rank() * team.team_size() + team.team_rank();
          if (i < N) {
            dy(i) = a * dx(i) + dy(i);
          }
        });
    Kokkos::fence();
    Kokkos::deep_copy(hres, dy);
    verifySaxpy(a, hx, hy, hres, N);

    Kokkos::deep_copy(dy, hy);
    constexpr int k = 8;
    const int numBlocksK = (N + (blockSize * k) - 1) / (blockSize * k);

    Kokkos::parallel_for(
        "saxpy_blocks_threads_kops", Kokkos::TeamPolicy<>(numBlocksK, blockSize),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
          const int first =
              (team.league_rank() * team.team_size() + team.team_rank()) * k;
          for (int j = 0; j < k; j++) {
            const int i = first + j;
            if (i < N) {
              dy(i) = a * dx(i) + dy(i);
            }
          }
        });
    Kokkos::fence();
    Kokkos::deep_copy(hres, dy);
    verifySaxpy(a, hx, hy, hres, N);
  }
  Kokkos::finalize();
  return 0;
}
