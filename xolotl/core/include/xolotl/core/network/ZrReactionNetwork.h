#pragma once

#include <xolotl/core/network/ReactionNetwork.h>
#include <xolotl/core/network/ZrReaction.h>
#include <xolotl/core/network/ZrTraits.h>

namespace xolotl
{
namespace core
{
namespace network
{
namespace detail
{
class ZrReactionGenerator;

class ZrClusterUpdater;
} // namespace detail

class ZrReactionNetwork : public ReactionNetwork<ZrReactionNetwork>
{
	friend class ReactionNetwork<ZrReactionNetwork>;
	friend class detail::ReactionNetworkWorker<ZrReactionNetwork>;

public:
	using Superclass = ReactionNetwork<ZrReactionNetwork>;
	using Subpaving = typename Superclass::Subpaving;
	using Composition = typename Superclass::Composition;
	using Species = typename Superclass::Species;
	using IndexType = typename Superclass::IndexType;
	using ConcentrationsView = typename Superclass::ConcentrationsView;
	using FluxesView = typename Superclass::FluxesView;

	using Superclass::Superclass;

	IndexType
	checkLargestClusterId();

private:
	double
	checkLatticeParameter(double latticeParameter);

	double
	computeAtomicVolume(double latticeParameter)
	{
		// TODO: Define atomic volume used in dissociation rate
		return 0.0 * latticeParameter * latticeParameter * latticeParameter;
	}

	double
	checkImpurityRadius(double impurityRadius);

	detail::ZrReactionGenerator
	getReactionGenerator() const noexcept;
};

namespace detail
{
class ZrReactionGenerator :
	public ReactionGenerator<ZrReactionNetwork, ZrReactionGenerator>
{
	friend class ReactionGeneratorBase<ZrReactionNetwork, ZrReactionGenerator>;

public:
	using Network = ZrReactionNetwork;
	using Subpaving = typename Network::Subpaving;
	using Superclass =
		ReactionGenerator<ZrReactionNetwork, ZrReactionGenerator>;

	using Superclass::Superclass;

	template <typename TTag>
	KOKKOS_INLINE_FUNCTION
	void
	operator()(IndexType i, IndexType j, TTag tag) const;

	template <typename TTag>
	KOKKOS_INLINE_FUNCTION
	void
	addSinks(IndexType i, TTag tag) const;

private:
	ReactionCollection<Network>
	getReactionCollection() const;
};

class ZrClusterUpdater
{
public:
	using Network = ZrReactionNetwork;
	using ClusterData = typename Network::ClusterData;
	using IndexType = typename Network::IndexType;

	KOKKOS_INLINE_FUNCTION
	void
	updateDiffusionCoefficient(const ClusterData& data, IndexType clusterId,
		IndexType gridIndex) const;
};
} // namespace detail
} // namespace network
} // namespace core
} // namespace xolotl

#include <xolotl/core/network/ZrClusterGenerator.h>
