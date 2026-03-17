#include <Kokkos_Core.hpp>
#include <iostream>

int main() {
  Kokkos::initialize();
  {
    std::cout << "Kokkos execution space: "
              << Kokkos::DefaultExecutionSpace::name() << std::endl;

    const int configs[][2] = {{1, 64}, {2, 32}, {4, 16},
                              {8, 8},  {16, 4}, {32, 2}, {64, 1}};

    for (const auto& cfg : configs) {
      const int leagueSize = cfg[0];
      const int teamSize = cfg[1];

      Kokkos::parallel_for(
          "hello_team",
          Kokkos::TeamPolicy<>(leagueSize, teamSize),
          KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
            Kokkos::single(
                Kokkos::PerThread(team), [=]() {
                  Kokkos::printf("Hello from block %d/%d, thread %d/%d\\n",
                                 team.league_rank(), team.league_size(),
                                 team.team_rank(), team.team_size());
                });
          });
      Kokkos::fence();
    }
  }
  Kokkos::finalize();
  return 0;
}
