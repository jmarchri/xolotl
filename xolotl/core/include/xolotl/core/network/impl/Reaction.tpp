#pragma once

#include <algorithm>
#include <array>

#include <plsm/EnumIndexed.h>

#include <xolotl/core/Constants.h>
#include <xolotl/util/MathUtils.h>

namespace xolotl
{
namespace core
{
namespace network
{
template <typename TNetwork, typename TDerived>
KOKKOS_INLINE_FUNCTION
Reaction<TNetwork, TDerived>::Reaction(ReactionDataRef reactionData,
	ClusterDataRef clusterData, IndexType reactionId) :
	_clusterData(clusterData),
	_reactionId(reactionId),
	_rate(reactionData.getRates(reactionId)),
	_widths(reactionData.getWidths(reactionId)),
	_coefs(reactionData.getCoefficients(reactionId))
{
}

template <typename TNetwork, typename TDerived>
KOKKOS_INLINE_FUNCTION
void
Reaction<TNetwork, TDerived>::updateData(
	ReactionDataRef reactionData, ClusterDataRef clusterData)
{
	_clusterData = clusterData;
	_rate = reactionData.getRates(_reactionId);
}

template <typename TRegion>
KOKKOS_INLINE_FUNCTION
std::enable_if_t<(numberOfInterstitialSpecies<typename TRegion::EnumIndex>() >
					 1),
	plsm::DifferenceType<typename TRegion::ScalarType>>
getISizeForOverlap(const TRegion& singleClReg, const TRegion& pairCl1Reg,
	const TRegion& pairCl2Reg)
{
	using Species = typename TRegion::EnumIndex;
	return pairCl1Reg[Species::I].begin() +
		pairCl1Reg[Species::Perfect].begin() +
		pairCl1Reg[Species::Frank].begin() + pairCl2Reg[Species::I].begin() +
		pairCl2Reg[Species::Perfect].begin() +
		pairCl2Reg[Species::Frank].begin() -
		(singleClReg[Species::I].begin() +
			singleClReg[Species::Perfect].begin() +
			singleClReg[Species::Frank].begin());
}

template <typename TRegion>
KOKKOS_INLINE_FUNCTION
std::enable_if_t<numberOfInterstitialSpecies<typename TRegion::EnumIndex>() ==
		1,
	plsm::DifferenceType<typename TRegion::ScalarType>>
getISizeForOverlap(const TRegion& singleClReg, const TRegion& pairCl1Reg,
	const TRegion& pairCl2Reg)
{
	using Species = typename TRegion::EnumIndex;
	return pairCl1Reg[Species::I].begin() + pairCl2Reg[Species::I].begin() -
		singleClReg[Species::I].begin();
}

template <typename TRegion>
KOKKOS_INLINE_FUNCTION
std::enable_if_t<numberOfInterstitialSpecies<typename TRegion::EnumIndex>() ==
		0,
	plsm::DifferenceType<typename TRegion::ScalarType>>
getISizeForOverlap(const TRegion& singleClReg, const TRegion& pairCl1Reg,
	const TRegion& pairCl2Reg)
{
	return 0;
}

template <typename TRegion>
KOKKOS_INLINE_FUNCTION
std::enable_if_t<(numberOfVacancySpecies<typename TRegion::EnumIndex>() > 1),
	plsm::DifferenceType<typename TRegion::ScalarType>>
getVWidthForOverlap(const TRegion& singleClReg, const TRegion& pairCl1Reg,
	const TRegion& pairCl2Reg)
{
	typename TRegion::ScalarType width = 0;
	using Species = typename TRegion::EnumIndex;
	bool singleIsV = (singleClReg.getOrigin().isOnAxis(Species::V) ||
			 singleClReg.getOrigin().isOnAxis(Species::Void) ||
			 singleClReg.getOrigin().isOnAxis(Species::Faulted)),
		 cl1IsV = (pairCl1Reg.getOrigin().isOnAxis(Species::V) ||
			 pairCl1Reg.getOrigin().isOnAxis(Species::Void) ||
			 pairCl1Reg.getOrigin().isOnAxis(Species::Faulted)),
		 cl2IsV = (pairCl2Reg.getOrigin().isOnAxis(Species::V) ||
			 pairCl2Reg.getOrigin().isOnAxis(Species::Void) ||
			 pairCl2Reg.getOrigin().isOnAxis(Species::Faulted));
	// Get the bounds of each cluster
	typename TRegion::ScalarType loSingle = singleClReg[Species::V].begin() +
		singleClReg[Species::Void].begin() +
		singleClReg[Species::Faulted].begin() +
		singleClReg[Species::I].begin() +
		singleClReg[Species::Perfect].begin() +
		singleClReg[Species::Frank].begin(),
								 hiSingle = singleClReg[Species::V].end() +
		singleClReg[Species::Void].end() + singleClReg[Species::Faulted].end() +
		singleClReg[Species::I].end() + singleClReg[Species::Perfect].end() +
		singleClReg[Species::Frank].end() - 6,
								 loCl1 = pairCl1Reg[Species::V].begin() +
		pairCl1Reg[Species::Void].begin() +
		pairCl1Reg[Species::Faulted].begin() + pairCl1Reg[Species::I].begin() +
		pairCl1Reg[Species::Perfect].begin() +
		pairCl1Reg[Species::Frank].begin(),
								 hiCl1 = pairCl1Reg[Species::V].end() +
		pairCl1Reg[Species::Void].end() + pairCl1Reg[Species::Faulted].end() +
		pairCl1Reg[Species::I].end() + pairCl1Reg[Species::Perfect].end() +
		pairCl1Reg[Species::Frank].end() - 6,
								 loCl2 = pairCl2Reg[Species::V].begin() +
		pairCl2Reg[Species::Void].begin() +
		pairCl2Reg[Species::Faulted].begin() + pairCl2Reg[Species::I].begin() +
		pairCl2Reg[Species::Perfect].begin() +
		pairCl2Reg[Species::Frank].begin(),
								 hiCl2 = pairCl2Reg[Species::V].end() +
		pairCl2Reg[Species::Void].end() + pairCl2Reg[Species::Faulted].end() +
		pairCl2Reg[Species::I].end() + pairCl2Reg[Species::Perfect].end() +
		pairCl2Reg[Species::Frank].end() - 6;

	// All clusters are the same type
	if ((singleIsV && cl1IsV && cl2IsV) || (!singleIsV && !cl1IsV && !cl2IsV)) {
		for (typename TRegion::ScalarType j = loCl1; j <= hiCl1; j++) {
			width += util::min(hiSingle, hiCl2 + j) -
				util::max(loSingle, loCl2 + j) + 1;
		}
	}
	else if ((singleIsV && cl2IsV) || (!singleIsV && !cl2IsV)) {
		for (typename TRegion::ScalarType j = loCl1; j <= hiCl1; j++) {
			width += util::min(hiSingle, hiCl2 - j) -
				util::max(loSingle, loCl2 - j) + 1;
		}
	}
	else {
		for (typename TRegion::ScalarType j = loCl2; j <= hiCl2; j++) {
			width += util::min(hiSingle, hiCl1 - j) -
				util::max(loSingle, loCl1 - j) + 1;
		}
	}

	return width;
}

template <typename TRegion>
KOKKOS_INLINE_FUNCTION
std::enable_if_t<numberOfVacancySpecies<typename TRegion::EnumIndex>() == 1,
	plsm::DifferenceType<typename TRegion::ScalarType>>
getVWidthForOverlap(const TRegion& singleClReg, const TRegion& pairCl1Reg,
	const TRegion& pairCl2Reg)
{
	typename TRegion::ScalarType width = 0;
	using Species = typename TRegion::EnumIndex;
	auto iSize = getISizeForOverlap(singleClReg, pairCl1Reg, pairCl2Reg);
	for (auto j : makeIntervalRange(pairCl1Reg[Species::V])) {
		auto tempWidth = util::min(singleClReg[Species::V].end() - 1.0,
							 pairCl2Reg[Species::V].end() - 1.0 + j - iSize) -
			util::max(singleClReg[Species::V].begin(),
				pairCl2Reg[Species::V].begin() + j - iSize) +
			1.0;
		if (iSize > 0 && singleClReg[Species::V].end() - 1 > 0) {
			width += tempWidth;
		}
		else {
			width += util::max(0.0, tempWidth);
		}
	}
	return width;
}

template <typename TRegion>
KOKKOS_INLINE_FUNCTION
std::enable_if_t<numberOfVacancySpecies<typename TRegion::EnumIndex>() == 0,
	plsm::DifferenceType<typename TRegion::ScalarType>>
getVWidthForOverlap(const TRegion& singleClReg, const TRegion& pairCl1Reg,
	const TRegion& pairCl2Reg)
{
	return 0;
}

template <typename TNetwork, typename TDerived>
KOKKOS_INLINE_FUNCTION
typename Reaction<TNetwork, TDerived>::AmountType
Reaction<TNetwork, TDerived>::computeOverlap(const Region& singleClReg,
	const Region& pairCl1Reg, const Region& pairCl2Reg)
{
	constexpr auto speciesRangeNoI = NetworkType::getSpeciesRangeNoI();

	AmountType nOverlap = 1;
	for (auto i : speciesRangeNoI) {
		// The width is the subset of the tiles for which the
		// reaction is possible
		// For instance if we have X_1 + X_[3,5) ⇄ X_[5,7)
		// It is only possible for 4 within X_[3,5) and 5 within X_[5,7)
		// so the width is 1
		// More complicated with X_[3,5) + X_[5,7) ⇄ X_[9,11)
		// 3+6, 4+5, 4+6, width is 3

		// Special case for I
		if (isVacancy(i)) {
			_widths(i()) = static_cast<double>(
				getVWidthForOverlap(singleClReg, pairCl1Reg, pairCl2Reg));
		}
		else {
			// TODO: Would be nice to loop on the cluster with the smaller tile
			for (auto j : makeIntervalRange(pairCl1Reg[i])) {
				_widths(i()) += util::max(0.0,
					util::min(singleClReg[i].end() - 1.0,
						pairCl2Reg[i].end() - 1.0 + j) -
						util::max(
							singleClReg[i].begin(), pairCl2Reg[i].begin() + j) +
						1.0);
			}
		}
		nOverlap *= _widths(i());
	}

	//    if (nOverlap <= 0) {
	//        constexpr auto speciesRange = NetworkType::getSpeciesRange();
	//    	std::cout << "first reactant: ";
	//        for (auto i : speciesRange) {
	//        std::cout << pairCl1Reg[i].begin() << ", ";
	//        }
	//        std::cout << std::endl;
	//        for (auto i : speciesRange) {
	//        std::cout << pairCl1Reg[i].end() - 1 << ", ";
	//        }
	//        std::cout << std::endl << "second reactant: ";
	//        for (auto i : speciesRange) {
	//        std::cout << pairCl2Reg[i].begin() << ", ";
	//        }
	//        std::cout << std::endl;
	//        for (auto i : speciesRange) {
	//        std::cout << pairCl2Reg[i].end() - 1 << ", ";
	//        }
	//        std::cout << std::endl << "product: ";
	//        for (auto i : speciesRange) {
	//        std::cout << singleClReg[i].begin() << ", ";
	//        }
	//        std::cout << std::endl;
	//        for (auto i : speciesRange) {
	//        std::cout << singleClReg[i].end() - 1 << ", ";
	//        }
	//        std::cout << std::endl;
	//        std::cout << "Overlap: " << nOverlap << std::endl;
	//        std::cout << "Widths: ";
	//        for (auto i : speciesRangeNoI) {
	//        	std::cout << _widths(i()) << ", ";
	//        }
	//        std::cout << std::endl;
	//    }
	assert(nOverlap > 0);

	return nOverlap;
}

template <typename TNetwork, typename TDerived>
KOKKOS_INLINE_FUNCTION
ProductionReaction<TNetwork, TDerived>::ProductionReaction(
	ReactionDataRef reactionData, ClusterDataRef clusterData,
	IndexType reactionId, IndexType cluster0, IndexType cluster1,
	IndexType cluster2, IndexType cluster3) :
	Superclass(reactionData, clusterData, reactionId),
	_reactants({cluster0, cluster1}),
	_products({cluster2, cluster3})
{
	for (auto i : {0, 1}) {
		this->copyMomentIds(_reactants[i], _reactantMomentIds[i]);
		this->copyMomentIds(_products[i], _productMomentIds[i]);
	}

	this->initialize();
}

template <typename TNetwork, typename TDerived>
KOKKOS_INLINE_FUNCTION
ProductionReaction<TNetwork, TDerived>::ProductionReaction(
	ReactionDataRef reactionData, ClusterDataRef clusterData,
	IndexType reactionId, const detail::ClusterSet& clusterSet) :
	ProductionReaction(reactionData, clusterData, reactionId,
		clusterSet.cluster0, clusterSet.cluster1, clusterSet.cluster2,
		clusterSet.cluster3)
{
}

template <typename TNetwork, typename TDerived>
KOKKOS_INLINE_FUNCTION
void
ProductionReaction<TNetwork, TDerived>::computeCoefficients()
{
	// static
	const auto dummyRegion = Region(Composition{});

	// Find the overlap for this reaction
	constexpr auto speciesRangeNoI = NetworkType::getSpeciesRangeNoI();

	const auto& cl1Reg =
		this->_clusterData.getCluster(_reactants[0]).getRegion();
	const auto& cl2Reg =
		this->_clusterData.getCluster(_reactants[1]).getRegion();
	const auto& prod1Reg = (_products[0] == invalidIndex) ?
		dummyRegion :
		this->_clusterData.getCluster(_products[0]).getRegion();
	const auto& prod2Reg = (_products[1] == invalidIndex) ?
		dummyRegion :
		this->_clusterData.getCluster(_products[1]).getRegion();
	const auto& cl1Disp = cl1Reg.dispersion();
	const auto& cl2Disp = cl2Reg.dispersion();

	// If there is no product the overlap is 1
	double nOverlap = 1.0;
	// General case
	if (_products[0] != invalidIndex && _products[1] == invalidIndex)
		nOverlap =
			static_cast<double>(this->computeOverlap(prod1Reg, cl1Reg, cl2Reg));
	// Special case with two products
	else if (_products[0] != invalidIndex && _products[1] != invalidIndex) {
		// Combine the regions
		auto ilist = Kokkos::Array<plsm::Interval<AmountType>,
			NetworkType::getNumberOfSpecies()>();
		for (auto i : NetworkType::getSpeciesRange()) {
			auto inter = plsm::Interval<AmountType>(
				prod1Reg[i].begin() + prod2Reg[i].begin(),
				prod1Reg[i].end() + prod2Reg[i].end() - 1);
			ilist[i()] = inter;
		}
		auto prodReg = Region(ilist);
		nOverlap =
			static_cast<double>(this->computeOverlap(prodReg, cl1Reg, cl2Reg));
	}
	// No product case
	else {
		for (auto i : speciesRangeNoI) {
			this->_widths[i()] = 1.0;
		}
	}

	this->_coefs(0, 0, 0, 0) = nOverlap;
	for (auto i : speciesRangeNoI) {
		// First order sum on the first reactant
		auto factor = nOverlap / this->_widths[i()];
		for (double m : makeIntervalRange(prod2Reg[i]))
			for (double l : makeIntervalRange(cl2Reg[i])) {
				this->_coefs(i() + 1, 0, 0, 0) += factor *
					util::firstOrderSum(
						util::max(prod1Reg[i].begin() + m - l,
							static_cast<double>(cl1Reg[i].begin())),
						util::min(prod1Reg[i].end() - 1 + m - l,
							static_cast<double>(cl1Reg[i].end() - 1)),
						static_cast<double>(
							cl1Reg[i].end() - 1 + cl1Reg[i].begin()) /
							2.0);
			}

		this->_coefs(0, 0, 0, i() + 1) =
			this->_coefs(i() + 1, 0, 0, 0) / cl1Disp[i()];

		// First order sum on the second reactant
		for (double m : makeIntervalRange(prod2Reg[i]))
			for (double l : makeIntervalRange(cl1Reg[i])) {
				this->_coefs(0, i() + 1, 0, 0) += factor *
					util::firstOrderSum(
						util::max(prod1Reg[i].begin() + m - l,
							static_cast<double>(cl2Reg[i].begin())),
						util::min(prod1Reg[i].end() - 1 + m - l,
							static_cast<double>(cl2Reg[i].end() - 1)),
						static_cast<double>(
							cl2Reg[i].end() - 1 + cl2Reg[i].begin()) /
							2.0);
			}

		this->_coefs(0, 0, 1, i() + 1) =
			this->_coefs(0, i() + 1, 0, 0) / cl2Disp[i()];

		// Loop on the potential products
		for (auto p : {0, 1}) {
			auto prodId = _products[p];
			if (prodId == invalidIndex) {
				continue;
			}

			// Get the regions in the right order
			const auto& thisReg =
				(prodId == _products[0]) ? prod1Reg : prod2Reg;
			const auto& otherReg =
				(prodId == _products[0]) ? prod2Reg : prod1Reg;
			// Get the dispersion
			const auto& thisDispersion = thisReg.dispersion();

			// First order sum on the other product
			for (double m : makeIntervalRange(otherReg[i]))
				for (double l : makeIntervalRange(cl1Reg[i])) {
					this->_coefs(0, 0, p + 2, i() + 1) += factor *
						util::firstOrderSum( // p+2 because 0 and 1 are used for
											 // reactants
							util::max(static_cast<double>(thisReg[i].begin()),
								cl2Reg[i].begin() + l - m),
							util::min(static_cast<double>(thisReg[i].end() - 1),
								cl2Reg[i].end() - 1 + l - m),
							static_cast<double>(
								thisReg[i].end() - 1 + thisReg[i].begin()) /
								2.0);
				}
			this->_coefs(0, 0, p + 2, i() + 1) /= thisDispersion[i()];

			// Products first moments
			for (auto k : speciesRangeNoI) {
				// Second order sum
				if (k == i) {
					for (double m : makeIntervalRange(otherReg[i]))
						for (double l : makeIntervalRange(cl2Reg[i])) {
							this->_coefs(i() + 1, 0, p + 2, k() + 1) += factor *
								util::secondOrderOffsetSum(
									util::max(thisReg[i].begin() + m - l,
										static_cast<double>(cl1Reg[i].begin())),
									util::min(thisReg[i].end() - 1 + m - l,
										static_cast<double>(
											cl1Reg[i].end() - 1)),
									static_cast<double>(cl1Reg[i].end() - 1 +
										cl1Reg[i].begin()) /
										2.0,
									static_cast<double>(thisReg[i].end() - 1 +
										thisReg[i].begin()) /
										2.0,
									l - m);
						}
					this->_coefs(i() + 1, 0, p + 2, k() + 1) /=
						thisDispersion[k()];

					for (double m : makeIntervalRange(otherReg[i]))
						for (double l : makeIntervalRange(cl1Reg[i])) {
							this->_coefs(0, i() + 1, p + 2, k() + 1) += factor *
								util::secondOrderOffsetSum(
									util::max(thisReg[i].begin() + m - l,
										static_cast<double>(cl2Reg[i].begin())),
									util::min(thisReg[i].end() - 1 + m - l,
										static_cast<double>(
											cl2Reg[i].end() - 1)),
									static_cast<double>(cl2Reg[i].end() - 1 +
										cl2Reg[i].begin()) /
										2.0,
									static_cast<double>(thisReg[i].end() - 1 +
										thisReg[i].begin()) /
										2.0,
									l - m);
						}
					this->_coefs(0, i() + 1, p + 2, k() + 1) /=
						thisDispersion[k()];
				}
				else {
					this->_coefs(i() + 1, 0, p + 2, k() + 1) =
						this->_coefs(i() + 1, 0, 0, 0) *
						this->_coefs(0, 0, p + 2, k() + 1) / nOverlap;

					this->_coefs(0, i() + 1, p + 2, k() + 1) =
						this->_coefs(0, i() + 1, 0, 0) *
						this->_coefs(0, 0, p + 2, k() + 1) / nOverlap;
				}
			}
		}
	}

	for (auto i : speciesRangeNoI) {
		auto factor = nOverlap / this->_widths[i()];

		// First reactant first moments
		for (auto k : speciesRangeNoI) {
			if (k == i) {
				for (double m : makeIntervalRange(prod2Reg[i]))
					for (double l : makeIntervalRange(cl2Reg[i])) {
						this->_coefs(i() + 1, 0, 0, k() + 1) += factor *
							util::secondOrderSum(
								util::max(prod1Reg[i].begin() + m - l,
									static_cast<double>(cl1Reg[i].begin())),
								util::min(prod1Reg[i].end() - 1 + m - l,
									static_cast<double>(cl1Reg[i].end() - 1)),
								static_cast<double>(
									cl1Reg[i].end() - 1 + cl1Reg[i].begin()) /
									2.0);
					}
				this->_coefs(i() + 1, 0, 0, k() + 1) /= cl1Disp[k()];
			}
			else {
				this->_coefs(i() + 1, 0, 0, k() + 1) =
					this->_coefs(i() + 1, 0, 0, 0) *
					this->_coefs(k() + 1, 0, 0, 0) / (nOverlap * cl1Disp[k()]);
			}

			this->_coefs(0, i() + 1, 0, k() + 1) =
				this->_coefs(k() + 1, i() + 1, 0, 0) / cl1Disp[k()];
		}

		// Second reactant partial derivatives
		for (auto k : speciesRangeNoI) {
			if (k == i) {
				for (double m : makeIntervalRange(prod2Reg[i]))
					for (double l : makeIntervalRange(cl1Reg[i])) {
						this->_coefs(0, i() + 1, 1, k() + 1) += factor *
							util::secondOrderSum(
								util::max(prod1Reg[i].begin() + m - l,
									static_cast<double>(cl2Reg[i].begin())),
								util::min(prod1Reg[i].end() - 1 + m - l,
									static_cast<double>(cl2Reg[i].end() - 1)),
								static_cast<double>(
									cl2Reg[i].end() - 1 + cl2Reg[i].begin()) /
									2.0);
					}
				this->_coefs(0, i() + 1, 1, k() + 1) /= cl2Disp[k()];
			}
			else {
				this->_coefs(0, i() + 1, 1, k() + 1) =
					this->_coefs(0, i() + 1, 0, 0) *
					this->_coefs(0, k() + 1, 0, 0) / (nOverlap * cl2Disp[k()]);
			}

			this->_coefs(i() + 1, 0, 1, k() + 1) =
				this->_coefs(i() + 1, k() + 1, 0, 0) / cl2Disp[k()];
		}
	}

	// Now we loop over the 2 dimensions of the coefs to compute all
	// the possible sums over distances for the flux
	for (auto i : speciesRangeNoI) {
		auto factor = nOverlap / this->_widths[i()];
		for (auto j : speciesRangeNoI) {
			// Second order sum
			if (i == j) {
				for (double m : makeIntervalRange(prod2Reg[j]))
					for (double l : makeIntervalRange(cl1Reg[j])) {
						this->_coefs(i() + 1, j() + 1, 0, 0) +=
							(l -
								static_cast<double>(
									cl1Reg[j].end() - 1 + cl1Reg[j].begin()) /
									2.0) *
							factor *
							util::firstOrderSum(
								util::max(prod1Reg[j].begin() + m - l,
									static_cast<double>(cl2Reg[j].begin())),
								util::min(prod1Reg[j].end() - 1 + m - l,
									static_cast<double>(cl2Reg[j].end() - 1)),
								static_cast<double>(
									cl2Reg[j].end() - 1 + cl2Reg[j].begin()) /
									2.0);
					}
			}
			else {
				this->_coefs(i() + 1, j() + 1, 0, 0) =
					this->_coefs(i() + 1, 0, 0, 0) *
					this->_coefs(0, j() + 1, 0, 0) / nOverlap;
			}

			// Now we deal with the coefficients needed for the
			// first moments
			// Let's start with the products
			for (auto p : {0, 1}) {
				auto prodId = _products[p];
				if (prodId == invalidIndex) {
					continue;
				}

				// Get the regions in the right order
				const auto& thisReg =
					(prodId == _products[0]) ? prod1Reg : prod2Reg;
				const auto& otherReg =
					(prodId == _products[0]) ? prod2Reg : prod1Reg;
				// Get the dispersion
				const auto& thisDispersion = thisReg.dispersion();

				for (auto k : speciesRangeNoI) {
					// Third order sum
					if (i == j && j == k) {
						for (double m : makeIntervalRange(otherReg[i]))
							for (double l : makeIntervalRange(cl1Reg[i])) {
								this->_coefs(
									i() + 1, j() + 1, p + 2, k() + 1) +=
									(l -
										static_cast<double>(cl1Reg[i].end() -
											1 + cl1Reg[i].begin()) /
											2.0) *
									factor *
									util::secondOrderOffsetSum(
										util::max(thisReg[i].begin() + m - l,
											static_cast<double>(
												cl2Reg[i].begin())),
										util::min(thisReg[i].end() - 1 + m - l,
											static_cast<double>(
												cl2Reg[i].end() - 1)),
										static_cast<double>(cl2Reg[i].end() -
											1 + cl2Reg[i].begin()) /
											2.0,
										static_cast<double>(thisReg[i].end() -
											1 + thisReg[i].begin()) /
											2.0,
										l - m);
							}
						this->_coefs(i() + 1, j() + 1, p + 2, k() + 1) /=
							thisDispersion[k()];
					}
					else if (j == k) {
						this->_coefs(i() + 1, j() + 1, p + 2, k() + 1) =
							this->_coefs(i() + 1, 0, 0, 0) *
							this->_coefs(0, j() + 1, p + 2, k() + 1) / nOverlap;
					}
					else if (i == k) {
						this->_coefs(i() + 1, j() + 1, p + 2, k() + 1) =
							this->_coefs(0, j() + 1, 0, 0) *
							this->_coefs(i() + 1, 0, p + 2, k() + 1) / nOverlap;
					}
					else {
						this->_coefs(i() + 1, j() + 1, p + 2, k() + 1) =
							this->_coefs(i() + 1, 0, 0, 0) *
							this->_coefs(0, j() + 1, 0, 0) *
							this->_coefs(0, 0, p + 2, k() + 1) /
							(nOverlap * nOverlap);
					}
				}
			}

			// Let's take care of the first reactant first moments
			for (auto k : speciesRangeNoI) {
				// Third order sum
				if (i == j && j == k) {
					for (double m : makeIntervalRange(prod2Reg[i]))
						for (double l : makeIntervalRange(cl1Reg[i])) {
							this->_coefs(i() + 1, j() + 1, 0, k() + 1) +=
								(l -
									static_cast<double>(cl1Reg[i].end() - 1 +
										cl1Reg[i].begin()) /
										2.0) *
								(l -
									static_cast<double>(cl1Reg[i].end() - 1 +
										cl1Reg[i].begin()) /
										2.0) *
								factor *
								util::firstOrderSum(
									util::max(prod1Reg[i].begin() + m - l,
										static_cast<double>(cl2Reg[i].begin())),
									util::min(prod1Reg[i].end() - 1 + m - l,
										static_cast<double>(
											cl2Reg[i].end() - 1)),
									static_cast<double>(cl2Reg[i].end() - 1 +
										cl2Reg[i].begin()) /
										2.0);
						}
					this->_coefs(i() + 1, j() + 1, 0, k() + 1) /= cl1Disp[k()];
				}
				else if (i == k) {
					this->_coefs(i() + 1, j() + 1, 0, k() + 1) =
						this->_coefs(0, j() + 1, 0, 0) *
						this->_coefs(i() + 1, 0, 0, k() + 1) / nOverlap;
				}
				else if (j == k) {
					this->_coefs(i() + 1, j() + 1, 0, k() + 1) =
						this->_coefs(i() + 1, 0, 0, 0) *
						this->_coefs(0, j() + 1, 0, k() + 1) / nOverlap;
				}
				else {
					this->_coefs(i() + 1, j() + 1, 0, k() + 1) =
						this->_coefs(i() + 1, 0, 0, 0) *
						this->_coefs(0, j() + 1, 0, 0) *
						this->_coefs(k() + 1, 0, 0, 0) /
						(nOverlap * nOverlap * cl1Disp[k()]);
				}
			}

			// Let's take care of the second reactant partial derivatives
			for (auto k : speciesRangeNoI) {
				// Third order sum
				if (i == j && j == k) {
					for (double m : makeIntervalRange(prod2Reg[i]))
						for (double l : makeIntervalRange(cl2Reg[i])) {
							this->_coefs(i() + 1, j() + 1, 1, k() + 1) +=
								(l -
									static_cast<double>(cl2Reg[i].end() - 1 +
										cl2Reg[i].begin()) /
										2.0) *
								(l -
									static_cast<double>(cl2Reg[i].end() - 1 +
										cl2Reg[i].begin()) /
										2.0) *
								factor *
								util::firstOrderSum(
									util::max(prod1Reg[i].begin() + m - l,
										static_cast<double>(cl1Reg[i].begin())),
									util::min(prod1Reg[i].end() - 1 + m - l,
										static_cast<double>(
											cl1Reg[i].end() - 1)),
									(double)(cl1Reg[i].end() - 1 +
										cl1Reg[i].begin()) /
										2.0);
						}
					this->_coefs(i() + 1, j() + 1, 1, k() + 1) /= cl2Disp[k()];
				}
				else if (i == k) {
					this->_coefs(i() + 1, j() + 1, 1, k() + 1) =
						this->_coefs(0, j() + 1, 0, 0) *
						this->_coefs(i() + 1, 0, 1, k() + 1) / nOverlap;
				}
				else if (j == k) {
					this->_coefs(i() + 1, j() + 1, 1, k() + 1) =
						this->_coefs(i() + 1, 0, 0, 0) *
						this->_coefs(0, j() + 1, 1, k() + 1) / nOverlap;
				}
				else {
					this->_coefs(i() + 1, j() + 1, 1, k() + 1) =
						this->_coefs(i() + 1, 0, 0, 0) *
						this->_coefs(0, j() + 1, 0, 0) *
						this->_coefs(0, k() + 1, 0, 0) /
						(nOverlap * nOverlap * cl2Disp[k()]);
				}
			}
		}
	}
}

template <typename TRegion>
KOKKOS_INLINE_FUNCTION
std::enable_if_t<(numberOfVacancySpecies<typename TRegion::EnumIndex>() > 1),
	double>
getRateForProduction(const TRegion& pairCl0Reg, const TRegion& pairCl1Reg,
	const double r0, const double r1, const double dc0, const double dc1)
{
	constexpr double pi = ::xolotl::core::pi;
	constexpr double rCore = ::xolotl::core::alloyCoreRadius;
	const double zs = 4.0 * pi * (r0 + r1 + rCore);
	using Species = typename TRegion::EnumIndex;
	bool cl0IsSphere = (pairCl0Reg.getOrigin().isOnAxis(Species::V) ||
			 pairCl0Reg.getOrigin().isOnAxis(Species::Void) ||
			 pairCl0Reg.getOrigin().isOnAxis(Species::I)),
		 cl1IsSphere = (pairCl1Reg.getOrigin().isOnAxis(Species::V) ||
			 pairCl1Reg.getOrigin().isOnAxis(Species::Void) ||
			 pairCl1Reg.getOrigin().isOnAxis(Species::I));

	// Simple case
	if (cl0IsSphere && cl1IsSphere) {
		return zs * (dc0 + dc1);
	}

	double p = 0.0, zl = 0.0;
	if (r0 < r1) {
		p = 1.0 / (1.0 + pow(r1 / (3.0 * (r0 + rCore)), 2.0));
		zl = 4.0 * pow(pi, 2.0) * r1 / log(1.0 + 8.0 * r1 / (r0 + rCore));
	}
	else {
		p = 1.0 / (1.0 + pow(r0 / (3.0 * (r1 + rCore)), 2.0));
		zl = 4.0 * pow(pi, 2.0) * r0 / log(1.0 + 8.0 * r0 / (r1 + rCore));
	}

	double k_plus = (dc0 + dc1) * (p * zs + (1.0 - p) * zl);
	double bias = 1.0;
	if (pairCl0Reg.getOrigin().isOnAxis(Species::I) ||
		pairCl1Reg.getOrigin().isOnAxis(Species::I)) {
		bias = 1.2;
	}

	return k_plus * bias;
}

template <typename TRegion>
KOKKOS_INLINE_FUNCTION
std::enable_if_t<(numberOfVacancySpecies<typename TRegion::EnumIndex>() < 2),
	double>
getRateForProduction(const TRegion& pairCl0Reg, const TRegion& pairCl1Reg,
	const double r0, const double r1, const double dc0, const double dc1)
{
	constexpr double pi = ::xolotl::core::pi;

	double kPlus = 4.0 * pi * (r0 + r1) * (dc0 + dc1);

	return kPlus;
}

template <typename TNetwork, typename TDerived>
KOKKOS_INLINE_FUNCTION
double
ProductionReaction<TNetwork, TDerived>::computeRate(IndexType gridIndex)
{
	auto cl0 = this->_clusterData.getCluster(_reactants[0]);
	auto cl1 = this->_clusterData.getCluster(_reactants[1]);

	double r0 = cl0.getReactionRadius();
	double r1 = cl1.getReactionRadius();

	double dc0 = cl0.getDiffusionCoefficient(gridIndex);
	double dc1 = cl1.getDiffusionCoefficient(gridIndex);

	return getRateForProduction(
		cl0.getRegion(), cl1.getRegion(), r0, r1, dc0, dc1);
}

template <typename TNetwork, typename TDerived>
KOKKOS_INLINE_FUNCTION
void
ProductionReaction<TNetwork, TDerived>::computeConnectivity(
	const Connectivity& connectivity)
{
	constexpr auto speciesRangeNoI = NetworkType::getSpeciesRangeNoI();
	// Get the total number of elements in each cluster
	auto cl1 = this->_clusterData.getCluster(_reactants[0]);
	const auto& cl1Reg = cl1.getRegion();
	const bool cl1IsSimplex = cl1Reg.isSimplex();
	auto cl2 = this->_clusterData.getCluster(_reactants[1]);
	const auto& cl2Reg = cl2.getRegion();
	const bool cl2IsSimplex = cl2Reg.isSimplex();
	// Each reactant connects with all the reactants
	// Reactant 1 with reactant 1
	this->addConnectivity(_reactants[0], _reactants[0], connectivity);
	if (!cl1IsSimplex) {
		for (auto i : speciesRangeNoI) {
			this->addConnectivity(
				_reactants[0], _reactantMomentIds[0][i()], connectivity);
			this->addConnectivity(
				_reactantMomentIds[0][i()], _reactants[0], connectivity);
			for (auto j : speciesRangeNoI) {
				this->addConnectivity(_reactantMomentIds[0][i()],
					_reactantMomentIds[0][j()], connectivity);
			}
		}
	}
	// Reactant 2 with reactant 1
	this->addConnectivity(_reactants[1], _reactants[0], connectivity);
	if (!cl1IsSimplex) {
		for (auto i : speciesRangeNoI) {
			this->addConnectivity(
				_reactants[1], _reactantMomentIds[0][i()], connectivity);
		}
	}
	if (!cl2IsSimplex) {
		for (auto i : speciesRangeNoI) {
			this->addConnectivity(
				_reactantMomentIds[1][i()], _reactants[0], connectivity);
		}
	}
	if (!cl1IsSimplex && !cl2IsSimplex) {
		for (auto i : speciesRangeNoI) {
			for (auto j : speciesRangeNoI) {
				this->addConnectivity(_reactantMomentIds[1][i()],
					_reactantMomentIds[0][j()], connectivity);
			}
		}
	}
	// Reactant 1 with reactant 2
	this->addConnectivity(_reactants[0], _reactants[1], connectivity);
	if (!cl2IsSimplex) {
		for (auto i : speciesRangeNoI) {
			this->addConnectivity(
				_reactants[0], _reactantMomentIds[1][i()], connectivity);
		}
	}
	if (!cl1IsSimplex) {
		for (auto i : speciesRangeNoI) {
			this->addConnectivity(
				_reactantMomentIds[0][i()], _reactants[1], connectivity);
		}
	}
	if (!cl1IsSimplex && !cl2IsSimplex) {
		for (auto i : speciesRangeNoI) {
			for (auto j : speciesRangeNoI) {
				this->addConnectivity(_reactantMomentIds[0][i()],
					_reactantMomentIds[1][j()], connectivity);
			}
		}
	}
	// Reactant 2 with reactant 2
	this->addConnectivity(_reactants[1], _reactants[1], connectivity);
	if (!cl2IsSimplex) {
		for (auto i : speciesRangeNoI) {
			this->addConnectivity(
				_reactants[1], _reactantMomentIds[1][i()], connectivity);
			this->addConnectivity(
				_reactantMomentIds[1][i()], _reactants[1], connectivity);
			for (auto j : speciesRangeNoI) {
				this->addConnectivity(_reactantMomentIds[1][i()],
					_reactantMomentIds[1][j()], connectivity);
			}
		}
	}
	// Each product connects with all the reactants
	for (auto p : {0, 1}) {
		auto prodId = _products[p];
		if (prodId == invalidIndex) {
			continue;
		}
		auto prod = this->_clusterData.getCluster(prodId);
		const auto& prodReg = prod.getRegion();
		const bool prodIsSimplex = prodReg.isSimplex();

		// With reactant 1
		this->addConnectivity(prodId, _reactants[0], connectivity);
		if (!cl1IsSimplex) {
			for (auto i : speciesRangeNoI) {
				this->addConnectivity(
					prodId, _reactantMomentIds[0][i()], connectivity);
			}
		}
		if (!prodIsSimplex) {
			for (auto i : speciesRangeNoI) {
				this->addConnectivity(
					_productMomentIds[p][i()], _reactants[0], connectivity);
			}
		}
		if (!cl1IsSimplex && !prodIsSimplex) {
			for (auto i : speciesRangeNoI) {
				for (auto j : speciesRangeNoI) {
					this->addConnectivity(_productMomentIds[p][i()],
						_reactantMomentIds[0][j()], connectivity);
				}
			}
		}
		// With reactant 2
		this->addConnectivity(prodId, _reactants[1], connectivity);
		if (!cl2IsSimplex) {
			for (auto i : speciesRangeNoI) {
				this->addConnectivity(
					prodId, _reactantMomentIds[1][i()], connectivity);
			}
		}
		if (!prodIsSimplex) {
			for (auto i : speciesRangeNoI) {
				this->addConnectivity(
					_productMomentIds[p][i()], _reactants[1], connectivity);
			}
		}
		if (!cl2IsSimplex && !prodIsSimplex) {
			for (auto i : speciesRangeNoI) {
				for (auto j : speciesRangeNoI) {
					this->addConnectivity(_productMomentIds[p][i()],
						_reactantMomentIds[1][j()], connectivity);
				}
			}
		}
	}
}

template <typename TNetwork, typename TDerived>
KOKKOS_INLINE_FUNCTION
void
ProductionReaction<TNetwork, TDerived>::computeFlux(
	ConcentrationsView concentrations, FluxesView fluxes, IndexType gridIndex)
{
	constexpr auto speciesRangeNoI = NetworkType::getSpeciesRangeNoI();

	// Compute the total number of elements in each cluster
	auto cl1 = this->_clusterData.getCluster(_reactants[0]);
	const auto& cl1Reg = cl1.getRegion();
	AmountType volCl1 = cl1Reg.volume();
	auto cl2 = this->_clusterData.getCluster(_reactants[1]);
	const auto& cl2Reg = cl2.getRegion();
	AmountType volCl2 = cl2Reg.volume();

	// Compute the flux for the 0th order moments
	double f = this->_coefs(0, 0, 0, 0) * concentrations[_reactants[0]] *
		concentrations[_reactants[1]];
	for (auto i : speciesRangeNoI) {
		f += this->_coefs(i() + 1, 0, 0, 0) *
			concentrations[_reactantMomentIds[0][i()]] *
			concentrations[_reactants[1]];
	}
	for (auto j : speciesRangeNoI) {
		f += this->_coefs(0, j() + 1, 0, 0) * concentrations[_reactants[0]] *
			concentrations[_reactantMomentIds[1][j()]];
	}
	for (auto i : speciesRangeNoI) {
		for (auto j : speciesRangeNoI) {
			f += this->_coefs(i() + 1, j() + 1, 0, 0) *
				concentrations[_reactantMomentIds[0][i()]] *
				concentrations[_reactantMomentIds[1][j()]];
		}
	}
	f *= this->_rate(gridIndex);

	Kokkos::atomic_sub(&fluxes[_reactants[0]], f / (double)volCl1);
	Kokkos::atomic_sub(&fluxes[_reactants[1]], f / (double)volCl2);
	for (auto prodId : _products) {
		if (prodId == invalidIndex) {
			continue;
		}

		auto prod = this->_clusterData.getCluster(prodId);
		const auto& prodReg = prod.getRegion();
		AmountType volProd = prodReg.volume();
		Kokkos::atomic_add(&fluxes[prodId], f / (double)volProd);
	}

	// Take care of the first moments
	for (auto k : speciesRangeNoI) {
		// First for the first reactant
		if (volCl1 > 1) {
			f = this->_coefs(0, 0, 0, k() + 1) * concentrations[_reactants[0]] *
				concentrations[_reactants[1]];
			for (auto i : speciesRangeNoI) {
				f += this->_coefs(i() + 1, 0, 0, k() + 1) *
					concentrations[_reactantMomentIds[0][i()]] *
					concentrations[_reactants[1]];
			}
			for (auto j : speciesRangeNoI) {
				f += this->_coefs(0, j() + 1, 0, k() + 1) *
					concentrations[_reactants[0]] *
					concentrations[_reactantMomentIds[1][j()]];
			}
			for (auto i : speciesRangeNoI) {
				for (auto j : speciesRangeNoI) {
					f += this->_coefs(i() + 1, j() + 1, 0, k() + 1) *
						concentrations[_reactantMomentIds[0][i()]] *
						concentrations[_reactantMomentIds[1][j()]];
				}
			}
			f *= this->_rate(gridIndex);
			Kokkos::atomic_sub(
				&fluxes[_reactantMomentIds[0][k()]], f / (double)volCl1);
		}

		// For the second reactant
		if (volCl2 > 1) {
			f = this->_coefs(0, 0, 1, k() + 1) * concentrations[_reactants[0]] *
				concentrations[_reactants[1]];
			for (auto i : speciesRangeNoI) {
				f += this->_coefs(i() + 1, 0, 1, k() + 1) *
					concentrations[_reactantMomentIds[0][i()]] *
					concentrations[_reactants[1]];
			}
			for (auto j : speciesRangeNoI) {
				f += this->_coefs(0, j() + 1, 1, k() + 1) *
					concentrations[_reactants[0]] *
					concentrations[_reactantMomentIds[1][j()]];
			}
			for (auto i : speciesRangeNoI) {
				for (auto j : speciesRangeNoI) {
					f += this->_coefs(i() + 1, j() + 1, 1, k() + 1) *
						concentrations[_reactantMomentIds[0][i()]] *
						concentrations[_reactantMomentIds[1][j()]];
				}
			}
			f *= this->_rate(gridIndex);
			Kokkos::atomic_sub(
				&fluxes[_reactantMomentIds[1][k()]], f / (double)volCl2);
		}

		// For the products
		for (auto p : {0, 1}) {
			auto prodId = _products[p];
			if (prodId == invalidIndex) {
				continue;
			}

			auto prod = this->_clusterData.getCluster(prodId);
			const auto& prodReg = prod.getRegion();
			AmountType volProd = prodReg.volume();

			if (volProd > 1) {
				f = this->_coefs(0, 0, p + 2, k() + 1) *
					concentrations[_reactants[0]] *
					concentrations[_reactants[1]];
				for (auto i : speciesRangeNoI) {
					f += this->_coefs(i() + 1, 0, p + 2, k() + 1) *
						concentrations[_reactantMomentIds[0][i()]] *
						concentrations[_reactants[1]];
				}
				for (auto j : speciesRangeNoI) {
					f += this->_coefs(0, j() + 1, p + 2, k() + 1) *
						concentrations[_reactants[0]] *
						concentrations[_reactantMomentIds[1][j()]];
				}
				for (auto i : speciesRangeNoI) {
					for (auto j : speciesRangeNoI) {
						f += this->_coefs(i() + 1, j() + 1, p + 2, k() + 1) *
							concentrations[_reactantMomentIds[0][i()]] *
							concentrations[_reactantMomentIds[1][j()]];
					}
				}
				f *= this->_rate(gridIndex);
				Kokkos::atomic_add(
					&fluxes[_productMomentIds[p][k()]], f / (double)volProd);
			}
		}
	}
}

template <typename TNetwork, typename TDerived>
KOKKOS_INLINE_FUNCTION
void
ProductionReaction<TNetwork, TDerived>::computePartialDerivatives(
	ConcentrationsView concentrations, Kokkos::View<double*> values,
	Connectivity connectivity, IndexType gridIndex)
{
	constexpr auto speciesRangeNoI = NetworkType::getSpeciesRangeNoI();
	int nProd = 0;
	for (auto prodId : _products) {
		if (prodId != invalidIndex) {
			++nProd;
		}
	}

	// Compute the total number of elements in each cluster
	auto cl1 = this->_clusterData.getCluster(_reactants[0]);
	const auto& cl1Reg = cl1.getRegion();
	AmountType volCl1 = cl1Reg.volume();
	auto cl2 = this->_clusterData.getCluster(_reactants[1]);
	const auto& cl2Reg = cl2.getRegion();
	AmountType volCl2 = cl2Reg.volume();

	// Compute the partials for the 0th order moments
	// Compute the values (d / dL_0^A)
	double temp = this->_coefs(0, 0, 0, 0) * concentrations[_reactants[1]];
	if (volCl2 > 1) {
		for (auto i : speciesRangeNoI) {
			temp += this->_coefs(0, i() + 1, 0, 0) *
				concentrations[_reactantMomentIds[1][i()]];
		}
	}
	// First for the first reactant
	Kokkos::atomic_sub(&values(connectivity(_reactants[0], _reactants[0])),
		this->_rate(gridIndex) * temp / (double)volCl1);
	// Second reactant
	Kokkos::atomic_sub(&values(connectivity(_reactants[1], _reactants[0])),
		this->_rate(gridIndex) * temp / (double)volCl2);
	// For the products
	for (auto p : {0, 1}) {
		auto prodId = _products[p];
		if (prodId == invalidIndex) {
			continue;
		}
		auto prod = this->_clusterData.getCluster(prodId);
		const auto& prodReg = prod.getRegion();
		AmountType volProd = prodReg.volume();
		Kokkos::atomic_add(&values(connectivity(prodId, _reactants[0])),
			this->_rate(gridIndex) * temp / (double)volProd);
	}

	// Compute the values (d / dL_0^B)
	temp = this->_coefs(0, 0, 0, 0) * concentrations[_reactants[0]];
	if (volCl1 > 1) {
		for (auto i : speciesRangeNoI) {
			temp += this->_coefs(i() + 1, 0, 0, 0) *
				concentrations[_reactantMomentIds[0][i()]];
		}
	}
	// First for the first reactant
	Kokkos::atomic_sub(&values(connectivity(_reactants[0], _reactants[1])),
		this->_rate(gridIndex) * temp / (double)volCl1);
	// Second reactant
	Kokkos::atomic_sub(&values(connectivity(_reactants[1], _reactants[1])),
		this->_rate(gridIndex) * temp / (double)volCl2);
	// For the products
	for (auto p : {0, 1}) {
		auto prodId = _products[p];
		if (prodId == invalidIndex) {
			continue;
		}
		auto prod = this->_clusterData.getCluster(prodId);
		const auto& prodReg = prod.getRegion();
		AmountType volProd = prodReg.volume();
		Kokkos::atomic_add(&values(connectivity(prodId, _reactants[1])),
			this->_rate(gridIndex) * temp / (double)volProd);
	}

	// (d / dL_1^A)
	if (volCl1 > 1) {
		for (auto i : speciesRangeNoI) {
			temp =
				this->_coefs(i() + 1, 0, 0, 0) * concentrations[_reactants[1]];
			if (volCl2 > 1) {
				for (auto j : speciesRangeNoI) {
					temp += this->_coefs(i() + 1, j() + 1, 0, 0) *
						concentrations[_reactantMomentIds[1][j()]];
				}
			}
			// First reactant
			Kokkos::atomic_sub(&values(connectivity(
								   _reactants[0], _reactantMomentIds[0][i()])),
				this->_rate(gridIndex) * temp / (double)volCl1);
			// second reactant
			Kokkos::atomic_sub(&values(connectivity(
								   _reactants[1], _reactantMomentIds[0][i()])),
				this->_rate(gridIndex) * temp / (double)volCl2);
			// For the products
			for (auto p : {0, 1}) {
				auto prodId = _products[p];
				if (prodId == invalidIndex) {
					continue;
				}
				auto prod = this->_clusterData.getCluster(prodId);
				const auto& prodReg = prod.getRegion();
				AmountType volProd = prodReg.volume();
				Kokkos::atomic_add(
					&values(connectivity(prodId, _reactantMomentIds[0][i()])),
					this->_rate(gridIndex) * temp / (double)volProd);
			}
		}
	}

	// (d / dL_1^B)
	if (volCl2 > 1) {
		for (auto i : speciesRangeNoI) {
			temp =
				this->_coefs(0, i() + 1, 0, 0) * concentrations[_reactants[0]];
			if (volCl1 > 1) {
				for (auto j : speciesRangeNoI) {
					temp += this->_coefs(j() + 1, i() + 1, 0, 0) *
						concentrations[_reactantMomentIds[0][j()]];
				}
			}
			Kokkos::atomic_sub(&values(connectivity(
								   _reactants[0], _reactantMomentIds[1][i()])),
				this->_rate(gridIndex) * temp / (double)volCl1);
			Kokkos::atomic_sub(&values(connectivity(
								   _reactants[1], _reactantMomentIds[1][i()])),
				this->_rate(gridIndex) * temp / (double)volCl2);
			for (auto p : {0, 1}) {
				auto prodId = _products[p];
				if (prodId == invalidIndex) {
					continue;
				}
				auto prod = this->_clusterData.getCluster(prodId);
				const auto& prodReg = prod.getRegion();
				AmountType volProd = prodReg.volume();
				Kokkos::atomic_add(
					&values(connectivity(prodId, _reactantMomentIds[1][i()])),
					this->_rate(gridIndex) * temp / (double)volProd);
			}
		}
	}

	// Take care of the first moments
	if (volCl1 > 1) {
		for (auto k : speciesRangeNoI) {
			// First for the first reactant
			// (d / dL_0^A)
			temp =
				this->_coefs(0, 0, 0, k() + 1) * concentrations[_reactants[1]];
			if (volCl2 > 1) {
				for (auto j : speciesRangeNoI) {
					temp += this->_coefs(0, j() + 1, 0, k() + 1) *
						concentrations[_reactantMomentIds[1][j()]];
				}
			}
			Kokkos::atomic_sub(&values(connectivity(
								   _reactantMomentIds[0][k()], _reactants[0])),
				this->_rate(gridIndex) * temp / (double)volCl1);
			// (d / dL_1^A)
			for (auto i : speciesRangeNoI) {
				temp = this->_coefs(i() + 1, 0, 0, k() + 1) *
					concentrations[_reactants[1]];
				if (volCl2 > 1) {
					for (auto j : speciesRangeNoI) {
						temp += this->_coefs(i() + 1, j() + 1, 0, k() + 1) *
							concentrations[_reactantMomentIds[1][j()]];
					}
				}
				Kokkos::atomic_sub(
					&values(connectivity(_reactantMomentIds[0][k()],
						_reactantMomentIds[0][i()])),
					this->_rate(gridIndex) * temp / (double)volCl1);
			}
			// (d / dL_0^B)
			temp =
				this->_coefs(0, 0, 0, k() + 1) * concentrations[_reactants[0]];
			for (auto j : speciesRangeNoI) {
				temp += this->_coefs(j() + 1, 0, 0, k() + 1) *
					concentrations[_reactantMomentIds[0][j()]];
			}
			Kokkos::atomic_sub(&values(connectivity(
								   _reactantMomentIds[0][k()], _reactants[1])),
				this->_rate(gridIndex) * temp / (double)volCl1);
			// (d / dL_1^B)
			if (volCl2 > 1) {
				for (auto i : speciesRangeNoI) {
					temp = this->_coefs(0, i() + 1, 0, k() + 1) *
						concentrations[_reactants[0]];
					for (auto j : speciesRangeNoI) {
						temp += this->_coefs(j() + 1, i() + 1, 0, k() + 1) *
							concentrations[_reactantMomentIds[0][j()]];
					}
					Kokkos::atomic_sub(
						&values(connectivity(_reactantMomentIds[0][k()],
							_reactantMomentIds[1][i()])),
						this->_rate(gridIndex) * temp / (double)volCl1);
				}
			}
		}
	}

	// Take care of the first moments
	if (volCl2 > 1) {
		for (auto k : speciesRangeNoI) {
			// First for the second reactant
			// (d / dL_0^A)
			temp =
				this->_coefs(0, 0, 1, k() + 1) * concentrations[_reactants[1]];
			for (auto j : speciesRangeNoI) {
				temp += this->_coefs(0, j() + 1, 1, k() + 1) *
					concentrations[_reactantMomentIds[1][j()]];
			}
			Kokkos::atomic_sub(&values(connectivity(
								   _reactantMomentIds[1][k()], _reactants[0])),
				this->_rate(gridIndex) * temp / (double)volCl2);
			// (d / dL_1^A)
			if (volCl1 > 1) {
				for (auto i : speciesRangeNoI) {
					temp = this->_coefs(i() + 1, 0, 1, k() + 1) *
						concentrations[_reactants[1]];
					for (auto j : speciesRangeNoI) {
						temp += this->_coefs(i() + 1, j() + 1, 1, k() + 1) *
							concentrations[_reactantMomentIds[1][j()]];
					}
					Kokkos::atomic_sub(
						&values(connectivity(_reactantMomentIds[1][k()],
							_reactantMomentIds[0][i()])),
						this->_rate(gridIndex) * temp / (double)volCl2);
				}
			}
			// (d / dL_0^B)
			temp =
				this->_coefs(0, 0, 1, k() + 1) * concentrations[_reactants[0]];
			if (volCl1 > 1) {
				for (auto j : speciesRangeNoI) {
					temp += this->_coefs(j() + 1, 0, 1, k() + 1) *
						concentrations[_reactantMomentIds[0][j()]];
				}
			}
			Kokkos::atomic_sub(&values(connectivity(
								   _reactantMomentIds[1][k()], _reactants[1])),
				this->_rate(gridIndex) * temp / (double)volCl2);
			// (d / dL_1^B)
			for (auto i : speciesRangeNoI) {
				temp = this->_coefs(0, i() + 1, 1, k() + 1) *
					concentrations[_reactants[0]];
				if (volCl1 > 1) {
					for (auto j : speciesRangeNoI) {
						temp += this->_coefs(j() + 1, i() + 1, 1, k() + 1) *
							concentrations[_reactantMomentIds[0][j()]];
					}
				}
				Kokkos::atomic_sub(
					&values(connectivity(_reactantMomentIds[1][k()],
						_reactantMomentIds[1][i()])),
					this->_rate(gridIndex) * temp / (double)volCl2);
			}
		}
	}

	// Loop on the products
	for (auto p : {0, 1}) {
		auto prodId = _products[p];
		if (prodId == invalidIndex) {
			continue;
		}

		auto prod = this->_clusterData.getCluster(prodId);
		const auto& prodReg = prod.getRegion();
		AmountType volProd = prodReg.volume();

		// Take care of the first moments
		if (volProd > 1) {
			for (auto k : speciesRangeNoI) {
				// (d / dL_0^A)
				temp = this->_coefs(0, 0, p + 2, k() + 1) *
					concentrations[_reactants[1]];
				if (volCl2 > 1) {
					for (auto j : speciesRangeNoI) {
						temp += this->_coefs(0, j() + 1, p + 2, k() + 1) *
							concentrations[_reactantMomentIds[1][j()]];
					}
				}
				Kokkos::atomic_add(
					&values(
						connectivity(_productMomentIds[p][k()], _reactants[0])),
					this->_rate(gridIndex) * temp / (double)volProd);
				// (d / dL_1^A)
				if (volCl1 > 1) {
					for (auto i : speciesRangeNoI) {
						temp = this->_coefs(i() + 1, 0, p + 2, k() + 1) *
							concentrations[_reactants[1]];
						if (volCl2 > 1) {
							for (auto j : speciesRangeNoI) {
								temp += this->_coefs(
											i() + 1, j() + 1, p + 2, k() + 1) *
									concentrations[_reactantMomentIds[1][j()]];
							}
						}
						Kokkos::atomic_add(
							&values(connectivity(_productMomentIds[p][k()],
								_reactantMomentIds[0][i()])),
							this->_rate(gridIndex) * temp / (double)volProd);
					}
				}
				// (d / dL_0^B)
				temp = this->_coefs(0, 0, p + 2, k() + 1) *
					concentrations[_reactants[0]];
				if (volCl1 > 1) {
					for (auto j : speciesRangeNoI) {
						temp += this->_coefs(j() + 1, 0, p + 2, k() + 1) *
							concentrations[_reactantMomentIds[0][j()]];
					}
				}
				Kokkos::atomic_add(
					&values(
						connectivity(_productMomentIds[p][k()], _reactants[1])),
					this->_rate(gridIndex) * temp / (double)volProd);
				// (d / dL_1^B)
				if (volCl2 > 1) {
					for (auto i : speciesRangeNoI) {
						temp = this->_coefs(0, i() + 1, p + 2, k() + 1) *
							concentrations[_reactants[0]];
						if (volCl1 > 1) {
							for (auto j : speciesRangeNoI) {
								temp += this->_coefs(
											j() + 1, i() + 1, p + 2, k() + 1) *
									concentrations[_reactantMomentIds[0][j()]];
							}
						}
						Kokkos::atomic_add(
							&values(connectivity(_productMomentIds[p][k()],
								_reactantMomentIds[1][i()])),
							this->_rate(gridIndex) * temp / (double)volProd);
					}
				}
			}
		}
	}
}

template <typename TNetwork, typename TDerived>
KOKKOS_INLINE_FUNCTION
double
ProductionReaction<TNetwork, TDerived>::computeLeftSideRate(
	ConcentrationsView concentrations, IndexType clusterId, IndexType gridIndex)
{
	// Check if our cluster is on the left side of this reaction
	if (clusterId == _reactants[0]) {
		return this->_rate(gridIndex) * concentrations[_reactants[1]] *
			this->_coefs(0, 0, 0, 0);
	}
	if (clusterId == _reactants[1]) {
		return this->_rate(gridIndex) * concentrations[_reactants[0]] *
			this->_coefs(0, 0, 0, 0);
	}

	// This cluster is not part of the reaction
	return 0.0;
}

template <typename TNetwork, typename TDerived>
KOKKOS_INLINE_FUNCTION
DissociationReaction<TNetwork, TDerived>::DissociationReaction(
	ReactionDataRef reactionData, ClusterDataRef clusterData,
	IndexType reactionId, IndexType cluster0, IndexType cluster1,
	IndexType cluster2) :
	Superclass(reactionData, clusterData, reactionId),
	_reactant(cluster0),
	_products({cluster1, cluster2})
{
	this->copyMomentIds(_reactant, _reactantMomentIds);
	for (auto i : {0, 1}) {
		this->copyMomentIds(_products[i], _productMomentIds[i]);
	}

	this->initialize();
}

template <typename TNetwork, typename TDerived>
KOKKOS_INLINE_FUNCTION
DissociationReaction<TNetwork, TDerived>::DissociationReaction(
	ReactionDataRef reactionData, ClusterDataRef clusterData,
	IndexType reactionId, const detail::ClusterSet& clusterSet) :
	DissociationReaction(reactionData, clusterData, reactionId,
		clusterSet.cluster0, clusterSet.cluster1, clusterSet.cluster2)
{
}

template <typename TNetwork, typename TDerived>
KOKKOS_INLINE_FUNCTION
void
DissociationReaction<TNetwork, TDerived>::computeCoefficients()
{
	constexpr auto speciesRangeNoI = NetworkType::getSpeciesRangeNoI();

	auto clReg = this->_clusterData.getCluster(_reactant).getRegion();
	auto prod1Reg = this->_clusterData.getCluster(_products[0]).getRegion();
	auto prod2Reg = this->_clusterData.getCluster(_products[1]).getRegion();
	const auto& clDisp = clReg.dispersion();
	const auto& prod1Disp = prod1Reg.dispersion();
	const auto& prod2Disp = prod2Reg.dispersion();

	auto nOverlap =
		static_cast<double>(this->computeOverlap(clReg, prod1Reg, prod2Reg));

	// The first coefficient is simply the overlap because it is the sum over 1
	this->_coefs(0, 0, 0, 0) = nOverlap;
	for (auto i : speciesRangeNoI) {
		auto factor = nOverlap / this->_widths[i()];
		// First order sum
		for (double l : makeIntervalRange(prod1Reg[i])) {
			this->_coefs(i() + 1, 0, 0, 0) += factor *
				util::firstOrderSum(
					util::max(static_cast<double>(clReg[i].begin()),
						prod2Reg[i].begin() + l),
					util::min(static_cast<double>(clReg[i].end() - 1),
						prod2Reg[i].end() - 1 + l),
					static_cast<double>(clReg[i].end() - 1 + clReg[i].begin()) /
						2.0);
		}
	}

	// First moments
	for (auto k : speciesRangeNoI) {
		auto factor = nOverlap / this->_widths[k()];
		// Reactant
		this->_coefs(0, 0, 0, k() + 1) =
			this->_coefs(k() + 1, 0, 0, 0) / clDisp[k()];

		// First product
		for (double l : makeIntervalRange(prod2Reg[k])) {
			this->_coefs(0, 0, 1, k() + 1) += factor *
				util::firstOrderSum(
					util::max(clReg[k].begin() - l,
						static_cast<double>(prod1Reg[k].begin())),
					util::min(clReg[k].end() - 1 - l,
						static_cast<double>(prod1Reg[k].end() - 1)),
					static_cast<double>(
						prod1Reg[k].end() - 1 + prod1Reg[k].begin()) /
						2.0);
		}
		this->_coefs(0, 0, 1, k() + 1) /= prod1Disp[k()];

		// Second product
		for (double l : makeIntervalRange(prod1Reg[k])) {
			this->_coefs(0, 0, 2, k() + 1) += factor *
				util::firstOrderSum(
					util::max(clReg[k].begin() - l,
						static_cast<double>(prod2Reg[k].begin())),
					util::min(clReg[k].end() - 1 - l,
						static_cast<double>(prod2Reg[k].end() - 1)),
					static_cast<double>(
						prod2Reg[k].end() - 1 + prod2Reg[k].begin()) /
						2.0);
		}
		this->_coefs(0, 0, 2, k() + 1) /= prod2Disp[k()];
	}

	// Now we loop over the 1 dimension of the coefs to compute all the
	// possible sums over distances for the flux
	for (auto i : speciesRangeNoI) {
		auto factor = nOverlap / this->_widths[i()];
		// Now we deal with the coefficients needed for the partial derivatives
		// Starting with the reactant
		for (auto k : speciesRangeNoI) {
			// Second order sum
			if (k == i) {
				for (double l : makeIntervalRange(prod1Reg[i])) {
					this->_coefs(i() + 1, 0, 0, k() + 1) += factor *
						util::secondOrderSum(
							util::max(static_cast<double>(clReg[i].begin()),
								prod2Reg[i].begin() + l),
							util::min(static_cast<double>(clReg[i].end() - 1),
								prod2Reg[i].end() - 1 + l),
							static_cast<double>(
								clReg[i].end() - 1 + clReg[i].begin()) /
								2.0);
				}
				this->_coefs(i() + 1, 0, 0, k() + 1) /= clDisp[k()];
			}
			else {
				this->_coefs(i() + 1, 0, 0, k() + 1) =
					this->_coefs(i() + 1, 0, 0, 0) *
					this->_coefs(k() + 1, 0, 0, 0) / (nOverlap * clDisp[k()]);
			}
		}

		// First moments for the first product
		for (auto k : speciesRangeNoI) {
			// Second order sum
			if (k == i) {
				for (double l : makeIntervalRange(prod2Reg[i])) {
					this->_coefs(i() + 1, 0, 1, k() + 1) += factor *
						util::secondOrderOffsetSum(
							util::max(static_cast<double>(clReg[i].begin()),
								prod1Reg[i].begin() + l),
							util::min(static_cast<double>(clReg[i].end() - 1),
								prod1Reg[i].end() - 1 + l),
							static_cast<double>(
								clReg[i].end() - 1 + clReg[i].begin()) /
								2.0,
							static_cast<double>(
								prod1Reg[i].end() - 1 + prod1Reg[i].begin()) /
								2.0,
							-l);
				}
				this->_coefs(i() + 1, 0, 1, k() + 1) /= prod1Disp[k()];
			}
			else {
				this->_coefs(i() + 1, 0, 1, k() + 1) =
					this->_coefs(i() + 1, 0, 0, 0) *
					this->_coefs(0, 0, 1, k() + 1) / nOverlap;
			}
		}

		// First moments for the second product
		for (auto k : speciesRangeNoI) {
			// Second order sum
			if (k == i) {
				for (double l : makeIntervalRange(prod1Reg[i])) {
					this->_coefs(i() + 1, 0, 2, k() + 1) += factor *
						util::secondOrderOffsetSum(
							util::max(static_cast<double>(clReg[i].begin()),
								prod2Reg[i].begin() + l),
							util::min(static_cast<double>(clReg[i].end() - 1),
								prod2Reg[i].end() - 1 + l),
							static_cast<double>(
								clReg[i].end() - 1 + clReg[i].begin()) /
								2.0,
							static_cast<double>(
								prod2Reg[i].end() - 1 + prod2Reg[i].begin()) /
								2.0,
							-l);
				}
				this->_coefs(i() + 1, 0, 2, k() + 1) /= prod2Disp[k()];
			}
			else {
				this->_coefs(i() + 1, 0, 2, k() + 1) =
					this->_coefs(i() + 1, 0, 0, 0) *
					this->_coefs(0, 0, 2, k() + 1) / nOverlap;
			}
		}
	}
}

template <typename TNetwork, typename TDerived>
KOKKOS_INLINE_FUNCTION
double
DissociationReaction<TNetwork, TDerived>::computeRate(IndexType gridIndex)
{
	double omega = this->_clusterData.getAtomicVolume();
	double T = this->_clusterData.temperature(gridIndex);

	// TODO: computeProductionRate should use products and not reactants
	auto cl0 = this->_clusterData.getCluster(_products[0]);
	auto cl1 = this->_clusterData.getCluster(_products[1]);

	double r0 = cl0.getReactionRadius();
	double r1 = cl1.getReactionRadius();

	double dc0 = cl0.getDiffusionCoefficient(gridIndex);
	double dc1 = cl1.getDiffusionCoefficient(gridIndex);

	double kPlus = getRateForProduction(
		cl0.getRegion(), cl1.getRegion(), r0, r1, dc0, dc1);
	double E_b = this->asDerived()->computeBindingEnergy();

	constexpr double k_B = ::xolotl::core::kBoltzmann;

	double kMinus = (1.0 / omega) * kPlus * std::exp(-E_b / (k_B * T));

	return kMinus;
}

template <typename TNetwork, typename TDerived>
KOKKOS_INLINE_FUNCTION
void
DissociationReaction<TNetwork, TDerived>::computeConnectivity(
	const Connectivity& connectivity)
{
	constexpr auto speciesRangeNoI = NetworkType::getSpeciesRangeNoI();

	// Get the total number of elements in each cluster
	auto cl = this->_clusterData.getCluster(_reactant);
	const auto& clReg = cl.getRegion();
	const bool clIsSimplex = clReg.isSimplex();
	auto prod1 = this->_clusterData.getCluster(_products[0]);
	const auto& prod1Reg = prod1.getRegion();
	const bool prod1IsSimplex = prod1Reg.isSimplex();
	auto prod2 = this->_clusterData.getCluster(_products[1]);
	const auto& prod2Reg = prod2.getRegion();
	const bool prod2IsSimplex = prod2Reg.isSimplex();

	// The reactant connects with the reactant
	this->addConnectivity(_reactant, _reactant, connectivity);
	if (!clIsSimplex) {
		for (auto i : speciesRangeNoI) {
			this->addConnectivity(
				_reactant, _reactantMomentIds[i()], connectivity);
			this->addConnectivity(
				_reactantMomentIds[i()], _reactant, connectivity);
			for (auto j : speciesRangeNoI) {
				this->addConnectivity(_reactantMomentIds[i()],
					_reactantMomentIds[j()], connectivity);
			}
		}
	}
	// Each product connects with  the reactant
	// Product 1 with reactant
	this->addConnectivity(_products[0], _reactant, connectivity);
	if (!clIsSimplex) {
		for (auto i : speciesRangeNoI) {
			this->addConnectivity(
				_products[0], _reactantMomentIds[i()], connectivity);
		}
	}
	if (!prod1IsSimplex) {
		for (auto i : speciesRangeNoI) {
			this->addConnectivity(
				_productMomentIds[0][i()], _reactant, connectivity);
		}
	}
	if (!clIsSimplex && !prod1IsSimplex) {
		for (auto i : speciesRangeNoI) {
			for (auto j : speciesRangeNoI) {
				this->addConnectivity(_productMomentIds[0][i()],
					_reactantMomentIds[j()], connectivity);
			}
		}
	}
	// Product 2 with reactant
	this->addConnectivity(_products[1], _reactant, connectivity);
	if (!clIsSimplex) {
		for (auto i : speciesRangeNoI) {
			this->addConnectivity(
				_products[1], _reactantMomentIds[i()], connectivity);
		}
	}
	if (!prod2IsSimplex) {
		for (auto i : speciesRangeNoI) {
			this->addConnectivity(
				_productMomentIds[1][i()], _reactant, connectivity);
		}
	}
	if (!clIsSimplex && !prod2IsSimplex) {
		for (auto i : speciesRangeNoI) {
			for (auto j : speciesRangeNoI) {
				this->addConnectivity(_productMomentIds[1][i()],
					_reactantMomentIds[j()], connectivity);
			}
		}
	}
}

template <typename TNetwork, typename TDerived>
KOKKOS_INLINE_FUNCTION
void
DissociationReaction<TNetwork, TDerived>::computeFlux(
	ConcentrationsView concentrations, FluxesView fluxes, IndexType gridIndex)
{
	constexpr auto speciesRangeNoI = NetworkType::getSpeciesRangeNoI();

	// Compute the total number of elements in each cluster
	auto cl = this->_clusterData.getCluster(_reactant);
	const auto& clReg = cl.getRegion();
	AmountType volCl = clReg.volume();
	auto prod1 = this->_clusterData.getCluster(_products[0]);
	const auto& prod1Reg = prod1.getRegion();
	AmountType volProd1 = prod1Reg.volume();
	auto prod2 = this->_clusterData.getCluster(_products[1]);
	const auto& prod2Reg = prod2.getRegion();
	AmountType volProd2 = prod2Reg.volume();

	// Compute the flux for the 0th order moments
	double f = this->_coefs(0, 0, 0, 0) * concentrations[_reactant];
	for (auto i : speciesRangeNoI) {
		f += this->_coefs(i() + 1, 0, 0, 0) *
			concentrations[_reactantMomentIds[i()]];
	}
	f *= this->_rate(gridIndex);
	Kokkos::atomic_sub(&fluxes[_reactant], f / (double)volCl);
	Kokkos::atomic_add(&fluxes[_products[0]], f / (double)volProd1);
	Kokkos::atomic_add(&fluxes[_products[1]], f / (double)volProd2);

	// Take care of the first moments
	for (auto k : speciesRangeNoI) {
		// First for the reactant
		if (volCl > 1) {
			f = this->_coefs(0, 0, 0, k() + 1) * concentrations[_reactant];
			for (auto i : speciesRangeNoI) {
				f += this->_coefs(i() + 1, 0, 0, k() + 1) *
					concentrations[_reactantMomentIds[i()]];
			}
			f *= this->_rate(gridIndex);
			Kokkos::atomic_sub(
				&fluxes[_reactantMomentIds[k()]], f / (double)volCl);
		}

		// Now the first product
		if (volProd1 > 1) {
			f = this->_coefs(0, 0, 1, k() + 1) * concentrations[_reactant];
			for (auto i : speciesRangeNoI) {
				f += this->_coefs(i() + 1, 0, 1, k() + 1) *
					concentrations[_reactantMomentIds[i()]];
			}
			f *= this->_rate(gridIndex);
			Kokkos::atomic_add(
				&fluxes[_productMomentIds[0][k()]], f / (double)volProd1);
		}

		// Finally the second product
		if (volProd2 > 1) {
			f = this->_coefs(0, 0, 2, k() + 1) * concentrations[_reactant];
			for (auto i : speciesRangeNoI) {
				f += this->_coefs(i() + 1, 0, 2, k() + 1) *
					concentrations[_reactantMomentIds[i()]];
			}
			f *= this->_rate(gridIndex);
			Kokkos::atomic_add(
				&fluxes[_productMomentIds[1][k()]], f / (double)volProd2);
		}
	}
}

template <typename TNetwork, typename TDerived>
KOKKOS_INLINE_FUNCTION
void
DissociationReaction<TNetwork, TDerived>::computePartialDerivatives(
	ConcentrationsView concentrations, Kokkos::View<double*> values,
	Connectivity connectivity, IndexType gridIndex)
{
	using AmountType = typename NetworkType::AmountType;
	constexpr auto speciesRangeNoI = NetworkType::getSpeciesRangeNoI();

	// Compute the total number of elements in each cluster
	auto cl = this->_clusterData.getCluster(_reactant);
	const auto& clReg = cl.getRegion();
	AmountType volCl = clReg.volume();
	auto prod1 = this->_clusterData.getCluster(_products[0]);
	const auto& prod1Reg = prod1.getRegion();
	AmountType volProd1 = prod1Reg.volume();
	auto prod2 = this->_clusterData.getCluster(_products[1]);
	const auto& prod2Reg = prod2.getRegion();
	AmountType volProd2 = prod2Reg.volume();

	// Compute the partials for the 0th order moments
	// First for the reactant
	double df = this->_rate(gridIndex) / (double)volCl;
	// Compute the values
	Kokkos::atomic_sub(&values(connectivity(_reactant, _reactant)),
		df * this->_coefs(0, 0, 0, 0));
	if (volProd1 > 1) {
		for (auto i : speciesRangeNoI) {
			Kokkos::atomic_sub(
				&values(connectivity(_reactant, _reactantMomentIds[i()])),
				df * this->_coefs(i() + 1, 0, 0, 0));
		}
	}
	// For the first product
	df = this->_rate(gridIndex) / (double)volProd1;
	Kokkos::atomic_add(&values(connectivity(_products[0], _reactant)),
		df * this->_coefs(0, 0, 0, 0));

	if (volProd1 > 1) {
		for (auto i : speciesRangeNoI) {
			Kokkos::atomic_add(
				&values(connectivity(_products[0], _reactantMomentIds[i()])),
				df * this->_coefs(i() + 1, 0, 0, 0));
		}
	}
	// For the second product
	df = this->_rate(gridIndex) / (double)volProd2;
	Kokkos::atomic_add(&values(connectivity(_products[1], _reactant)),
		df * this->_coefs(0, 0, 0, 0));

	if (volProd1 > 1) {
		for (auto i : speciesRangeNoI) {
			Kokkos::atomic_add(
				&values(connectivity(_products[1], _reactantMomentIds[i()])),
				df * this->_coefs(i() + 1, 0, 0, 0));
		}
	}

	// Take care of the first moments
	for (auto k : speciesRangeNoI) {
		if (volCl > 1) {
			// First for the reactant
			df = this->_rate(gridIndex) / (double)volCl;
			// Compute the values
			Kokkos::atomic_sub(
				&values(connectivity(_reactantMomentIds[k()], _reactant)),
				df * this->_coefs(0, 0, 0, k() + 1));
			for (auto i : speciesRangeNoI) {
				Kokkos::atomic_sub(&values(connectivity(_reactantMomentIds[k()],
									   _reactantMomentIds[i()])),
					df * this->_coefs(i() + 1, 0, 0, k() + 1));
			}
		}
		// For the first product
		if (volProd1 > 1) {
			df = this->_rate(gridIndex) / (double)volProd1;
			Kokkos::atomic_add(
				&values(connectivity(_productMomentIds[0][k()], _reactant)),
				df * this->_coefs(0, 0, 1, k() + 1));
			for (auto i : speciesRangeNoI) {
				Kokkos::atomic_add(
					&values(connectivity(
						_productMomentIds[0][k()], _reactantMomentIds[i()])),
					df * this->_coefs(i() + 1, 0, 1, k() + 1));
			}
		}
		// For the second product
		if (volProd2 > 1) {
			df = this->_rate(gridIndex) / (double)volProd2;
			Kokkos::atomic_add(
				&values(connectivity(_productMomentIds[1][k()], _reactant)),
				df * this->_coefs(0, 0, 2, k() + 1));
			for (auto i : speciesRangeNoI) {
				Kokkos::atomic_add(
					&values(connectivity(
						_productMomentIds[1][k()], _reactantMomentIds[i()])),
					df * this->_coefs(i() + 1, 0, 2, k() + 1));
			}
		}
	}
}

template <typename TNetwork, typename TDerived>
KOKKOS_INLINE_FUNCTION
double
DissociationReaction<TNetwork, TDerived>::computeLeftSideRate(
	ConcentrationsView concentrations, IndexType clusterId, IndexType gridIndex)
{
	// Check if our cluster is on the left side of this reaction
	if (clusterId == _reactant) {
		return this->_rate(gridIndex) * this->_coefs(0, 0, 0, 0);
	}

	// This cluster is not part of the reaction
	return 0.0;
}
} // namespace network
} // namespace core
} // namespace xolotl