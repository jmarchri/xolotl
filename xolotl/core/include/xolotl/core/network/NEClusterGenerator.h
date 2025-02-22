#pragma once

#include <plsm/refine/Detector.h>

namespace xolotl
{
namespace core
{
namespace network
{
class NEClusterGenerator : public plsm::refine::Detector<NEClusterGenerator>
{
public:
	using Superclass = plsm::refine::Detector<NEClusterGenerator>;
	using NetworkType = NEReactionNetwork;
	using Species = NESpecies;

	template <typename PlsmContext>
	using Cluster = Cluster<NetworkType, PlsmContext>;

	using Region = typename NetworkType::Region;
	using Composition = typename NetworkType::Composition;
	using AmountType = typename NetworkType::AmountType;
	using BoolArray = typename Superclass::BoolVec<Region>;

	NEClusterGenerator(const options::IOptions& opts);

	NEClusterGenerator(const options::IOptions& opts, std::size_t refineDepth);

	KOKKOS_INLINE_FUNCTION
	bool
	refine(const Region& region, BoolArray& result) const;

	KOKKOS_INLINE_FUNCTION
	bool
	select(const Region& region) const;

	template <typename PlsmContext>
	KOKKOS_INLINE_FUNCTION
	double
	getFormationEnergy(const Cluster<PlsmContext>& cluster) const noexcept;

	template <typename PlsmContext>
	KOKKOS_INLINE_FUNCTION
	double
	getMigrationEnergy(const Cluster<PlsmContext>& cluster) const noexcept;

	template <typename PlsmContext>
	KOKKOS_INLINE_FUNCTION
	double
	getDiffusionFactor(const Cluster<PlsmContext>& cluster,
		double latticeParameter) const noexcept;

	template <typename PlsmContext>
	KOKKOS_INLINE_FUNCTION
	double
	getReactionRadius(const Cluster<PlsmContext>& cluster,
		double latticeParameter, double interstitialBias,
		double impurityRadius) const noexcept;

private:
	AmountType _maxXe;
	double _xeDiffusivity;
	bool _xeDiffusive;
	AmountType _groupingMin;
	AmountType _groupingWidth;
	double _density;
};
} // namespace network
} // namespace core
} // namespace xolotl
