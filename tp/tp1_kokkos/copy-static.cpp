#include <Kokkos_Core.hpp>
#include <iostream>

constexpr int N = 1024;

int main() {
  Kokkos::initialize();
  {
    float A[N];
    float B[N];

    for (int i = 0; i < N; i++) {
      A[i] = static_cast<float>(i);
      B[i] = 0.0f;
    }

    Kokkos::View<float[N], Kokkos::DefaultExecutionSpace> dA("dA");

    auto hA = Kokkos::create_mirror_view(dA);
    for (int i = 0; i < N; i++) {
      hA(i) = A[i];
    }
    Kokkos::deep_copy(dA, hA);

    auto hB = Kokkos::create_mirror_view(dA);
    Kokkos::deep_copy(hB, dA);
    for (int i = 0; i < N; i++) {
      B[i] = hB(i);
    }

    int i = 0;
    for (; i < N; i++) {
      if (A[i] != B[i]) {
        break;
      }
    }

    if (i < N) {
      std::cout << "The copy is incorrect!" << std::endl;
    } else {
      std::cout << "The copy is correct!" << std::endl;
    }
  }
  Kokkos::finalize();
  return 0;
}
