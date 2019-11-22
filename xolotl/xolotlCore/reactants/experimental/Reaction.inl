#pragma once

#include <algorithm>
#include <array>

#include <plsm/EnumIndexed.h>

#include <Constants.h>

namespace xolotlCore
{
namespace experimental
{
template <typename TImpl>
template <typename TDerived>
ReactionNetwork<TImpl>::Reaction<TDerived>::Reaction(NetworkType& network,
    std::size_t reactionId, Type reactionType, std::size_t cluster0,
    std::size_t cluster1, std::size_t cluster2, std::size_t cluster3)
    :
    _network(&network),
    _type(reactionType),
    _fluxFn(
        _type == Type::production ? &Reaction::productionFlux :
            &Reaction::dissociationFlux),
    _partialsFn(
        _type == Type::production ? &Reaction::productionPartialDerivatives :
            &Reaction::dissociationPartialDerivatives),
    _reactants(
        _type == Type::production ? Kokkos::Array<std::size_t, 2>( {cluster0,
            cluster1}) : Kokkos::Array<std::size_t, 2>( {cluster0, invalid})),
    _products(_type == Type::production ? Kokkos::Array<std::size_t, 2>( {
        cluster2, cluster3}) : Kokkos::Array<std::size_t, 2>( {cluster1,
        cluster2})),
    _rate(_network->getReactionRates(reactionId))
{
    for (std::size_t i : {0, 1}) {
        copyMomentIds(_reactants[i], _reactantMomentIds[i]);
        copyMomentIds(_products[i], _productMomentIds[i]);
    }

    constexpr auto coeffExtent = NetworkType::getNumberOfSpeciesNoI() + 1;
    if (_type == Type::production) {
        int nProd = 0;
        for (auto prodId : _products) {
            if (prodId != invalid) {
                ++nProd;
            }
        }
        auto nCl = 2 + nProd;
        _coefs = Kokkos::View<double****>(
            "Flux Coefficients", coeffExtent, coeffExtent, nCl, coeffExtent);
        computeProductionCoefficients();
    }
    else {
        _coefs = Kokkos::View<double****>(
            "Flux Coefficients", coeffExtent, 1, 3, coeffExtent);
        computeDissociationCoefficients();
    }

    updateRates();
}

template <typename TImpl>
template <typename TDerived>
inline void
ReactionNetwork<TImpl>::Reaction<TDerived>::updateRates()
{
    if (_type == Type::production) {
        computeProductionRates();
    }
    else {
        computeDissociationRates();
    }
}

template <typename TImpl>
template <typename TDerived>
inline typename ReactionNetwork<TImpl>::AmountType
ReactionNetwork<TImpl>::Reaction<TDerived>::computeOverlap(
    const Region& singleClReg, const Region& pairCl1Reg,
    const Region& pairCl2Reg)
{
    constexpr auto numSpeciesNoI = NetworkType::getNumberOfSpeciesNoI();
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

        AmountType width{};

        //TODO: Would be nice to loop on the cluster with the smaller tile
        for (auto j : makeIntervalRange(pairCl1Reg[i])) {
            width +=
                std::min(singleClReg[i].end() - 1, pairCl2Reg[i].end() - 1 + j)
                - std::max(singleClReg[i].begin(), pairCl2Reg[i].begin() + j)
                + 1;
        }

        nOverlap *= width;
    }

    assert(nOverlap > 0);

    return nOverlap;
}

template <typename TImpl>
template <typename TDerived>
inline void
ReactionNetwork<TImpl>::Reaction<TDerived>::computeProductionCoefficients()
{
    static const auto dummyRegion = Region(Composition{});

    // Find the overlap for this reaction
    constexpr auto numSpeciesNoI = NetworkType::getNumberOfSpeciesNoI();
    constexpr auto speciesRangeNoI = NetworkType::getSpeciesRangeNoI();

    const auto& cl1Reg = _network->getCluster(_reactants[0]).getRegion();
    const auto& cl2Reg = _network->getCluster(_reactants[1]).getRegion();
    const auto& prod1Reg = (_products[0] == invalid) ? dummyRegion :
        _network->getCluster(_products[0]).getRegion();
    const auto& prod2Reg = (_products[1] == invalid) ? dummyRegion :
        _network->getCluster(_products[1]).getRegion();

    auto nOverlap =
        static_cast<double>(computeOverlap(prod1Reg, cl1Reg, cl2Reg));

    _coefs(0, 0, 0, 0) = nOverlap;
    for (auto i : speciesRangeNoI) {
        // First order sum on the first reactant
        for (auto m : makeIntervalRange(prod2Reg[i]))
        for (auto l : makeIntervalRange(cl2Reg[i])) {
            _coefs(i() + 1, 0, 0, 0) += firstOrderSum(
                std::max(prod1Reg[i].begin() + m - l, cl1Reg[i].begin()),
                std::min(prod1Reg[i].end() - 1 + m - l, cl1Reg[i].end() - 1),
                static_cast<double>(cl1Reg[i].end() - 1 + cl1Reg[i].begin())
                    / 2.0);
        }

        // First order sum on the second reactant
        for (auto m : makeIntervalRange(prod2Reg[i]))
        for (auto l : makeIntervalRange(cl1Reg[i])) {
            _coefs(0, i() + 1, 0, 0) += firstOrderSum(
                std::max(prod1Reg[i].begin() + m - l, cl2Reg[i].begin()),
                std::min(prod1Reg[i].end() - 1 + m - l, cl2Reg[i].end() - 1),
                static_cast<double>(cl2Reg[i].end() - 1 + cl2Reg[i].begin())
                    / 2.0);
        }

        // Loop on the potential products
        for (std::size_t p : {0,1}) {
            auto prodId = _products[p];
            if (prodId == invalid) {
                continue;
            }

            // Get the regions in the right order
            const auto& thisReg = (prodId == _products[0]) ? prod1Reg : prod2Reg;
            const auto& otherReg = (prodId == _products[0]) ? prod2Reg : prod1Reg;

            // First order sum on the second products
            for (auto m : makeIntervalRange(otherReg[i]))
            for (auto l : makeIntervalRange(cl1Reg[i])) {
                _coefs(0, 0, p+2, i() + 1) += firstOrderSum( // p+2 because 0 and 1 are used for reactants
                    std::max(thisReg[i].begin(), cl2Reg[i].begin() + l - m),
                    std::min(thisReg[i].end() - 1, cl2Reg[i].end() - 1 + l - m),
                    static_cast<double>(thisReg[i].end() - 1 + thisReg[i].begin())
                        / 2.0);
            }

            // Products first moments
            for (auto k : speciesRangeNoI) {
                // Second order sum
                if (k == i) {
                    for (auto m : makeIntervalRange(otherReg[i]))
                    for (auto l : makeIntervalRange(cl2Reg[i])) {
                    _coefs(i() + 1, 0, p+2, k() + 1) += secondOrderOffsetSum(
                            std::max(thisReg[i].begin() + m - l, cl1Reg[i].begin()),
                            std::min(thisReg[i].end() - 1 + m - l, cl1Reg[i].end() - 1),
                            (double) (cl1Reg[i].end() - 1 + cl1Reg[i].begin())
                                / 2.0,
                            (double) (thisReg[i].end() - 1 + thisReg[i].begin())
                                / 2.0, l - m);
                    }
                    for (auto m : makeIntervalRange(otherReg[i]))
                    for (auto l : makeIntervalRange(cl1Reg[i])) {
                        _coefs(0, i() + 1, p+2, k() + 1) += secondOrderOffsetSum(
                            std::max(thisReg[i].begin() + m - l, cl2Reg[i].begin()),
                            std::min(thisReg[i].end() - 1 + m - l, cl2Reg[i].end() - 1),
                            (double) (cl2Reg[i].end() - 1 + cl2Reg[i].begin())
                                / 2.0,
                            (double) (thisReg[i].end() - 1 + thisReg[i].begin())
                                / 2.0, l - m);
                    }
                }
                else {
                    _coefs(i() + 1, 0, p+2, k() + 1) += _coefs(i() + 1, 0, 0, 0)
                        * _coefs(0, 0, p+2, k() + 1) / nOverlap;

                    _coefs(0, i() + 1, p+2, k() + 1) += _coefs(0, i() + 1, 0, 0)
                        * _coefs(0, 0, p+2, k() + 1) / nOverlap;
                }
            }
        }
    }

    for (auto i : speciesRangeNoI) {
        // First reactant first moments
        for (auto k : speciesRangeNoI) {
            _coefs(0, 0, 0, k() + 1) += _coefs(k() + 1, 0, 0, 0);

            if (k == i) {
                for (auto m : makeIntervalRange(prod2Reg[i]))
                for (auto l : makeIntervalRange(cl2Reg[i])) {
                    _coefs(i() + 1, 0, 0, k()) += secondOrderSum(
                        std::max(prod1Reg[i].begin() + m - l, cl1Reg[i].begin()),
                        std::min(prod1Reg[i].end() - 1 + m - l, cl1Reg[i].end() - 1),
                        (double) (cl1Reg[i].end() - 1 + cl1Reg[i].begin())
                            / 2.0);
                }
            }
            else {
                _coefs(i() + 1, 0, 0, k() + 1) += _coefs(i() + 1, 0, 0, 0)
                    * _coefs(k() + 1, 0, 0, 0) / nOverlap;
            }

            _coefs(0, i() + 1, 0, k() + 1) += _coefs(k() + 1, i() + 1, 0, 0);
        }

        // Second reactant partial derivatives
        for (auto k : speciesRangeNoI) {
            _coefs(0, 0, 1, k() + 1) += _coefs(0, k() + 1, 0, 0);

            if (k == i) {
                for (auto m : makeIntervalRange(prod2Reg[i]))
                for (auto l : makeIntervalRange(cl1Reg[i])) {
                    _coefs(0, i() + 1, 1, k() + 1) += secondOrderSum(
                        std::max(prod1Reg[i].begin() + m - l, cl2Reg[i].begin()),
                        std::min(prod1Reg[i].end() - 1 + m - l, cl2Reg[i].end() - 1),
                        (double) (cl2Reg[i].end() - 1 + cl2Reg[i].begin())
                            / 2.0);
                }
            }
            else {
                _coefs(0, i() + 1, 1, k() + 1) += _coefs(0, i() + 1, 0, 0)
                    * _coefs(0, k() + 1, 0, 0) / nOverlap;
            }

            _coefs(i() + 1, 0, 1, k() + 1) += _coefs(i() + 1, k() + 1, 0, 0);
        }
    }

    // Now we loop over the 2 dimensions of the coefs to compute all
    // the possible sums over distances for the flux
    for (auto i : speciesRangeNoI) {
        for (auto j : speciesRangeNoI) {
            // Second order sum
            if (i == j) {
                for (auto m : makeIntervalRange(prod2Reg[j]))
                for (auto l : makeIntervalRange(cl1Reg[j])) {
                    _coefs(i() + 1, j() + 1, 0, 0) += (l
                        - (double) (cl1Reg[j].end() - 1 + cl1Reg[j].begin())
                            / 2.0)
                        * firstOrderSum(
                            std::max(prod1Reg[j].begin() + m - l, cl2Reg[j].begin()),
                            std::min(prod1Reg[j].end() - 1 + m - l,
                                cl2Reg[j].end() - 1),
                            (double) (cl2Reg[j].end() - 1 + cl2Reg[j].begin())
                                / 2.0);
                }
            }
            else {
                _coefs(i() + 1, j() + 1, 0, 0) += _coefs(i() + 1, 0, 0, 0)
                    * _coefs(0, j() + 1, 0, 0) / nOverlap;
            }

            // Now we deal with the coefficients needed for the
            // first moments
            // Let's start with the products
            for (std::size_t p : {0,1}) {
                auto prodId = _products[p];
                if (prodId == invalid) {
                    continue;
                }

                // Get the regions in the right order
                const auto& thisReg = (prodId == _products[0]) ? prod1Reg : prod2Reg;
                const auto& otherReg = (prodId == _products[0]) ? prod2Reg : prod1Reg;

                for (auto k : speciesRangeNoI) {
                    // Third order sum
                    if (i == j && j == k) {
                        for (auto m : makeIntervalRange(otherReg[i]))
                        for (auto l : makeIntervalRange(cl1Reg[i])) {
                            _coefs(i() + 1, j() + 1, p+2, k() + 1) += (l
                                - (double) (cl1Reg[i].end() - 1 + cl1Reg[i].begin())
                                    / 2.0)
                                * secondOrderOffsetSum(
                                    std::max(thisReg[i].begin() + m - l,
                                        cl2Reg[i].begin()),
                                    std::min(thisReg[i].end() - 1 + m - l,
                                        cl2Reg[i].end() - 1),
                                    (double) (cl2Reg[i].end() - 1
                                        + cl2Reg[i].begin()) / 2.0,
                                    (double) (thisReg[i].end() - 1
                                        + thisReg[i].begin()) / 2.0, l - m);
                        }
                    }
                    else if (j == k) {
                        _coefs(i() + 1, j() + 1, p+2, k() + 1) += _coefs(i() + 1, 0,
                            0, 0) * _coefs(0, j() + 1, p+2, k() + 1) / nOverlap;
                    }
                    else if (i == k) {
                        _coefs(i() + 1, j() + 1, p+2, k() + 1) += _coefs(0, j() + 1,
                            0, 0) * _coefs(i() + 1, 0, p+2, k() + 1) / nOverlap;
                    }
                    else {
                        // TODO check this is the right formula, might be divided by nOverlap^2
                        _coefs(i() + 1, j() + 1, p+2, k() + 1) += _coefs(i() + 1, 0,
                            0, 0) * _coefs(0, j() + 1, 0, 0)
                            * _coefs(0, 0, p+2, k() + 1) / nOverlap;
                    }
                }
            }

            // Let's take care of the first reactant first moments
            for (auto k : speciesRangeNoI) {
                // Third order sum
                if (i == j && j == k) {
                    for (auto m : makeIntervalRange(prod2Reg[i]))
                    for (auto l : makeIntervalRange(cl1Reg[i])) {
                        _coefs(i() + 1, j() + 1, 0, k() + 1) += (l
                            - (double) (cl1Reg[i].end() - 1 + cl1Reg[i].begin())
                                / 2.0)
                            * (l
                                - (double) (cl1Reg[i].end() - 1
                                    + cl1Reg[i].begin()) / 2.0)
                            * firstOrderSum(
                                std::max(prod1Reg[i].begin() + m - l,
                                    cl2Reg[i].begin()),
                                std::min(prod1Reg[i].end() - 1 + m - l,
                                    cl2Reg[i].end() - 1),
                                (double) (cl2Reg[i].end() - 1
                                    + cl2Reg[i].begin()) / 2.0);
                    }
                }
                else if (i == k) {
                    _coefs(i() + 1, j() + 1, 0, k() + 1) += _coefs(0, j() + 1,
                        0, 0) * _coefs(i() + 1, 0, 0, k() + 1) / nOverlap;
                }
                else if (j == k) {
                    _coefs(i() + 1, j() + 1, 0, k() + 1) += _coefs(i() + 1, 0,
                        0, 0) * _coefs(0, j() + 1, 0, k() + 1) / nOverlap;
                }
                else {
                    // TODO check this is the right formula, might be divided by nOverlap^2
                    _coefs(i() + 1, j() + 1, 0, k() + 1) += _coefs(i() + 1, 0,
                        0, 0) * _coefs(0, j() + 1, 0, 0)
                        * _coefs(k() + 1, 0, 0, 0) / nOverlap;
                }
            }

            // Let's take care of the second reactant partial derivatives
            for (auto k : speciesRangeNoI) {
                // Third order sum
                if (i == j && j == k) {
                    for (auto m : makeIntervalRange(prod2Reg[i]))
                    for (auto l : makeIntervalRange(cl2Reg[i])) {
                        _coefs(i() + 1, j() + 1, 1, k() + 1) += (l
                            - (double) (cl2Reg[i].end() - 1 + cl2Reg[i].begin())
                                / 2.0)
                            * (l
                                - (double) (cl2Reg[i].end() - 1
                                    + cl2Reg[i].begin()) / 2.0)
                            * firstOrderSum(
                                std::max(prod1Reg[i].begin() + m - l,
                                    cl1Reg[i].begin()),
                                std::min(prod1Reg[i].end() - 1 + m - l,
                                    cl1Reg[i].end() - 1),
                                (double) (cl1Reg[i].end() - 1
                                    + cl1Reg[i].begin()) / 2.0);
                    }
                }
                else if (i == k) {
                    _coefs(i() + 1, j() + 1, 1, k() + 1) += _coefs(0, j() + 1,
                        0, 0) * _coefs(i() + 1, 0, 1, k() + 1) / nOverlap;
                }
                else if (j == k) {
                    _coefs(i() + 1, j() + 1, 1, k() + 1) += _coefs(i() + 1, 0,
                        0, 0) * _coefs(0, j() + 1, 1, k() + 1) / nOverlap;
                }
                else {
                    // TODO check this is the right formula, might be divided by nOverlap^2
                    _coefs(i() + 1, j() + 1, 1, k() + 1) += _coefs(i() + 1, 0,
                        0, 0) * _coefs(0, j() + 1, 0, 0)
                        * _coefs(0, k() + 1, 0, 0) / nOverlap;
                }
            }
        }
    }
}

template <typename TImpl>
template <typename TDerived>
inline void
ReactionNetwork<TImpl>::Reaction<TDerived>::computeDissociationCoefficients()
{
    constexpr auto numSpeciesNoI = NetworkType::getNumberOfSpeciesNoI();
    constexpr auto speciesRangeNoI = NetworkType::getSpeciesRangeNoI();

    auto clReg = _network->getCluster(_reactants[0]).getRegion();
    auto prod1Reg = _network->getCluster(_products[0]).getRegion();
    auto prod2Reg = _network->getCluster(_products[1]).getRegion();

    auto nOverlap =
        static_cast<double>(computeOverlap(clReg, prod1Reg, prod2Reg));

    // The first coefficient is simply the overlap because it is the sum over 1
    _coefs(0, 0, 0, 0) = nOverlap;
    for (auto i : speciesRangeNoI) {
        // First order sum
        for (auto l : makeIntervalRange(prod1Reg[i])) {
            _coefs(i() + 1, 0, 0, 0) += firstOrderSum(
                std::max(clReg[i - 1].begin(), prod2Reg[i - 1].begin() + l),
                std::min(clReg[i - 1].end() - 1, prod2Reg[i - 1].end() - 1 + l),
                (double) (clReg[i - 1].end() - 1 + clReg[i - 1].begin()) / 2.0);
        }
    }

    // First moments
    for (auto k : speciesRangeNoI) {
        // Reactant
        _coefs(0, 0, 0, k() + 1) += _coefs(k() + 1, 0, 0, 0);

        // First product
        for (auto l : makeIntervalRange(prod2Reg[k])) {
            _coefs(0, 0, 1, k() + 1) += firstOrderSum(
                std::max(clReg[k].begin() - l, prod1Reg[k].begin()),
                std::min(clReg[k].end() - 1 - l, prod1Reg[k].end() - 1),
                (double) (prod1Reg[k].end() - 1 + prod1Reg[k].begin()) / 2.0);
        }

        // Second product
        for (auto l : makeIntervalRange(prod1Reg[k])) {
            _coefs(0, 0, 2, k() + 1) += firstOrderSum(
                std::max(clReg[k].begin() - l, prod2Reg[k].begin()),
                std::min(clReg[k].end() - 1 - l, prod2Reg[k].end() - 1),
                (double) (prod2Reg[k].end() - 1 + prod2Reg[k].begin()) / 2.0);
        }
    }

    // Now we loop over the 1 dimension of the coefs to compute all the
    // possible sums over distances for the flux
    for (auto i : speciesRangeNoI) {
        // Now we deal with the coefficients needed for the partial derivatives
        // Starting with the reactant
        for (auto k : speciesRangeNoI) {
            // Second order sum
            if (k == i) {
                for (auto l : makeIntervalRange(prod1Reg[i])) {
                    _coefs(i() + 1, 0, 0, k() + 1) += secondOrderSum(
                        std::max(clReg[i].begin(), prod2Reg[i].begin() + l),
                        std::min(clReg[i].end() - 1, prod2Reg[i].end() - 1 + l),
                        (double) (clReg[i].end() - 1 + clReg[i].begin()) / 2.0);
                }
            }
            else {
                _coefs(i() + 1, 0, 0, k() + 1) += _coefs(i() + 1, 0, 0, 0)
                    * _coefs(k() + 1, 0, 0, 0) / nOverlap;
            }
        }

        // First moments for the first product
        for (auto k : speciesRangeNoI) {
            // Second order sum
            if (k == i) {
                for (auto l : makeIntervalRange(prod2Reg[i])) {
                    _coefs(i() + 1, 0, 1, k() + 1) += secondOrderOffsetSum(
                        std::max(clReg[i].begin(), prod1Reg[i].begin() + l),
                        std::min(clReg[i].end() - 1, prod1Reg[i].end() - 1 + l),
                        (double) (clReg[i].end() - 1 + clReg[i].begin()) / 2.0,
                        (double) (prod1Reg[i].end() - 1 + prod1Reg[i].begin())
                            / 2.0, -l);
                }
            }
            else {
                _coefs(i() + 1, 0, 1, k() + 1) += _coefs(i() + 1, 0, 0, 0)
                    * _coefs(0, 0, 1, k() + 1) / nOverlap;
            }
        }

        // First moments for the second product
        for (auto k : speciesRangeNoI) {
            // Second order sum
            if (k == i) {
                for (auto l : makeIntervalRange(prod1Reg[i])) {
                    _coefs(i() + 1, 0, 2, k() + 1) += secondOrderOffsetSum(
                        std::max(clReg[i].begin(), prod2Reg[i].begin() + l),
                        std::min(clReg[i].end() - 1, prod2Reg[i].end() - 1 + l),
                        (double) (clReg[i].end() - 1 + clReg[i].begin()) / 2.0,
                        (double) (prod2Reg[i].end() - 1 + prod2Reg[i].begin())
                            / 2.0, -l);
                }
            }
            else {
                _coefs(i() + 1, 0, 2, k() + 1) += _coefs(i() + 1, 0, 0, 0)
                    * _coefs(0, 0, 2, k() + 1) / nOverlap;
            }
        }
    }
}

template <typename TImpl>
template <typename TDerived>
inline double
ReactionNetwork<TImpl>::Reaction<TDerived>::computeProductionRate(
    std::size_t gridIndex)
{
    auto cl0 = _network->getCluster(_reactants[0]);
    auto cl1 = _network->getCluster(_reactants[1]);

    double r0 = cl0.getReactionRadius();
    double r1 = cl1.getReactionRadius();

    double dc0 = cl0.getDiffusionCoefficient(gridIndex);
    double dc1 = cl1.getDiffusionCoefficient(gridIndex);

    constexpr double pi = ::xolotlCore::pi;

    double kPlus = 4.0 * pi * (r0 + r1) * (dc0 + dc1);

    return kPlus;
}

template <typename TImpl>
template <typename TDerived>
inline double
ReactionNetwork<TImpl>::Reaction<TDerived>::computeDissociationRate(
    std::size_t gridIndex)
{
    double omega = _network->getAtomicVolume();
    double T = _network->getTemperature(gridIndex);

    double kPlus = asDerived()->computeProductionRate(gridIndex);
    double E_b = asDerived()->computeBindingEnergy();

    constexpr double k_B = ::xolotlCore::kBoltzmann;

    double kMinus = (1.0 / omega) * kPlus * std::exp(-E_b / (k_B * T));

    return kMinus;
}

template <typename TImpl>
template <typename TDerived>
inline void
ReactionNetwork<TImpl>::Reaction<TDerived>::productionFlux(
    ConcentrationsView concentrations, FluxesView fluxes, std::size_t gridIndex)
{
    constexpr auto numSpeciesNoI = NetworkType::getNumberOfSpeciesNoI();
    constexpr auto speciesRangeNoI = NetworkType::getSpeciesRangeNoI();

    // Compute the total number of elements in each cluster
    auto cl1 = _network->getCluster(_reactants[0]);
    const auto& cl1Reg = cl1.getRegion();
    AmountType volCl1 = cl1Reg.volume();
    auto cl2 = _network->getCluster(_reactants[1]);
    const auto& cl2Reg = cl2.getRegion();
    AmountType volCl2 = cl2Reg.volume();

    // Compute the flux for the 0th order moments
    double f = _coefs(0, 0, 0, 0) * concentrations[_reactants[0]] *
        concentrations[_reactants[1]];
    for (auto i : speciesRangeNoI) {
        f += _coefs(i() + 1, 0, 0, 0) *
            concentrations[_reactantMomentIds[0][i()]] *
            concentrations[_reactants[1]];
    }
    for (auto j : speciesRangeNoI) {
        f += _coefs(0, j() + 1, 0, 0) * concentrations[_reactants[0]] *
            concentrations[_reactantMomentIds[1][j()]];
    }
    for (auto i : speciesRangeNoI) {
        for (auto j : speciesRangeNoI) {
            f += _coefs(i() + 1, j() + 1, 0, 0) *
                concentrations[_reactantMomentIds[0][i()]] *
                concentrations[_reactantMomentIds[1][j()]];
        }
    }
    f *= _rate(gridIndex);

    fluxes[_reactants[0]] -= f / (double) volCl1;
    fluxes[_reactants[1]] -= f / (double) volCl2;
    for (auto prodId : _products) {
        if (prodId == invalid) {
            continue;
        }

        auto prod = _network->getCluster(prodId);
        const auto& prodReg = prod.getRegion();
        AmountType volProd = prodReg.volume();
        fluxes[prodId] += f / (double) volProd;
    }

    // Take care of the first moments
    for (auto k : speciesRangeNoI) {
        // First for the first reactant
        f = _coefs(0, 0, 0, k() + 1) * concentrations[_reactants[0]] *
            concentrations[_reactants[1]];
        for (auto i : speciesRangeNoI) {
            f += _coefs(i() + 1, 0, 0, k() + 1) *
                concentrations[_reactantMomentIds[0][i()]] *
                concentrations[_reactants[1]];
        }
        for (auto j : speciesRangeNoI) {
            f += _coefs(0, j() + 1, 0, k() + 1) *
                concentrations[_reactants[0]] *
                concentrations[_reactantMomentIds[1][j()]];
        }
        for (auto i : speciesRangeNoI) {
            for (auto j : speciesRangeNoI) {
                f += _coefs(i() + 1, j() + 1, 0, k() + 1) *
                    concentrations[_reactantMomentIds[0][i()]] *
                    concentrations[_reactantMomentIds[1][j()]];
            }
        }
        // TODO compute the prefactor related to the dispersion, it can be
        // moved to the coefs maybe
        double prefactor = 1.0;
        f *= _rate(gridIndex) * prefactor;
        fluxes[_reactantMomentIds[0][k()]] -= f / (double) volCl1;

        // For the second reactant
        f = _coefs(0, 0, 1, k() + 1) * concentrations[_reactants[0]] *
            concentrations[_reactants[1]];
        for (auto i : speciesRangeNoI) {
            f += _coefs(i() + 1, 0, 1, k() + 1) *
                concentrations[_reactantMomentIds[0][i()]] *
                concentrations[_reactants[1]];
        }
        for (auto j : speciesRangeNoI) {
            f += _coefs(0, j() + 1, 1, k() + 1) *
                concentrations[_reactants[0]] *
                concentrations[_reactantMomentIds[1][j()]];
        }
        for (auto i : speciesRangeNoI) {
            for (auto j : speciesRangeNoI) {
                f += _coefs(i() + 1, j() + 1, 1, k() + 1) *
                    concentrations[_reactantMomentIds[0][i()]] *
                    concentrations[_reactantMomentIds[1][j()]];
            }
        }
        // TODO compute the prefactor related to the dispersion, it can be
        // moved to the coefs maybe
        prefactor = 1.0;
        f *= _rate(gridIndex) * prefactor;
        fluxes[_reactantMomentIds[1][k()]] -= f / (double) volCl2;

        // For the products
        for (std::size_t p : {0,1}) {
            auto prodId = _products[p];
            if (prodId == invalid) {
                continue;
            }

            auto prod = _network->getCluster(prodId);
            const auto& prodReg = prod.getRegion();
            AmountType volProd = prodReg.volume();

            f = _coefs(0, 0, p+2, k() + 1) * concentrations[_reactants[0]] *
                concentrations[_reactants[1]];
            for (auto i : speciesRangeNoI) {
                f += _coefs(i() + 1, 0, p+2, k() + 1) *
                    concentrations[_reactantMomentIds[0][i()]] *
                    concentrations[_reactants[1]];
            }
            for (auto j : speciesRangeNoI) {
                f += _coefs(0, j() + 1, p+2, k() + 1) *
                    concentrations[_reactants[0]] *
                    concentrations[_reactantMomentIds[1][j()]];
            }
            for (auto i : speciesRangeNoI) {
                for (auto j : speciesRangeNoI) {
                    f += _coefs(i() + 1, j() + 1, p+2, k() + 1) *
                        concentrations[_reactantMomentIds[0][i()]] *
                        concentrations[_reactantMomentIds[1][j()]];
                }
            }
            // TODO compute the prefactor related to the dispersion, it can be
            // moved to the coefs maybe
            prefactor = 1.0;
            f *= _rate(gridIndex) * prefactor;
            fluxes[_productMomentIds[p][k()]] -= f / (double) volProd;
        }
    }
}

template <typename TImpl>
template <typename TDerived>
inline void
ReactionNetwork<TImpl>::Reaction<TDerived>::dissociationFlux(
    ConcentrationsView concentrations, FluxesView fluxes, std::size_t gridIndex)
{
    constexpr auto numSpeciesNoI = NetworkType::getNumberOfSpeciesNoI();
    constexpr auto speciesRangeNoI = NetworkType::getSpeciesRangeNoI();

    // Compute the total number of elements in each cluster
    auto cl = _network->getCluster(_reactants[0]);
    const auto& clReg = cl.getRegion();
    AmountType volCl = clReg.volume();
    auto prod1 = _network->getCluster(_products[0]);
    const auto& prod1Reg = prod1.getRegion();
    AmountType volProd1 = prod1Reg.volume();
    auto prod2 = _network->getCluster(_products[1]);
    const auto& prod2Reg = prod2.getRegion();
    AmountType volProd2 = prod2Reg.volume();

    // Compute the flux for the 0th order moments
    double f = _coefs(0, 0, 0, 0) * concentrations[_reactants[0]];
    for (auto i : speciesRangeNoI) {
        f += _coefs(i() + 1, 0, 0, 0) *
            concentrations[_reactantMomentIds[0][i()]];
    }
    f *= _rate(gridIndex);
    fluxes[_reactants[0]] -= f / (double) volCl;
    fluxes[_products[0]] += f / (double) volProd1;
    fluxes[_products[1]] += f / (double) volProd2;

    // Take care of the first moments
    for (auto k : speciesRangeNoI) {
        // First for the reactant
        f = _coefs(0, 0, 0, k() + 1) * concentrations[_reactants[0]];
        for (auto i : speciesRangeNoI) {
            f += _coefs(i() + 1, 0, 0, k() + 1) *
                concentrations[_reactantMomentIds[0][i()]];
        }
        // TODO compute the prefactor related to the dispersion, it can be
        // moved to the coefs maybe
        double prefactor = 1.0;
        f *= _rate(gridIndex) * prefactor;
        fluxes[_reactantMomentIds[0][k()]] -= f / (double) volCl;

        // Now the first product
        f = _coefs(0, 0, 1, k() + 1) * concentrations[_reactants[0]];
        for (auto i : speciesRangeNoI) {
            f += _coefs(i() + 1, 0, 1, k() + 1) *
                concentrations[_reactantMomentIds[0][i()]];
        }
        // TODO compute the prefactor related to the dispersion, it can be
        // moved to the coefs maybe
        prefactor = 1.0;
        f *= _rate(gridIndex) * prefactor;
        fluxes[_productMomentIds[0][k()]] += f / (double) volProd1;

        // Finally the second product
        f = _coefs(0, 0, 2, k() + 1) * concentrations[_reactants[0]];
        for (auto i : speciesRangeNoI) {
            f += _coefs(i() + 1, 0, 2, k() + 1) *
                concentrations[_reactantMomentIds[0][i()]];
        }
        // TODO compute the prefactor related to the dispersion, it can be
        // moved to the coefs maybe
        prefactor = 1.0;
        f *= _rate(gridIndex) * prefactor;
        fluxes[_productMomentIds[1][k()]] += f / (double) volProd2;
    }
}

template <typename TImpl>
template <typename TDerived>
inline void
ReactionNetwork<TImpl>::Reaction<TDerived>::productionPartialDerivatives(
    ConcentrationsView concentrations, Kokkos::View<std::size_t*> indices,
    Kokkos::View<double*> values, std::size_t gridIndex)
{
    constexpr auto numSpeciesNoI = NetworkType::getNumberOfSpeciesNoI();
    constexpr auto speciesRangeNoI = NetworkType::getSpeciesRangeNoI();
    int nProd = 0;
    for (auto prodId : _products) {
        if (prodId != invalid) {
            ++nProd;
        }
    }
    
    // TODO The is probably a better way to index everything

    // Compute the total number of elements in each cluster
    auto cl1 = _network->getCluster(_reactants[0]);
    const auto& cl1Reg = cl1.getRegion();
    AmountType volCl1 = cl1Reg.volume();
    auto cl2 = _network->getCluster(_reactants[1]);
    const auto& cl2Reg = cl2.getRegion();
    AmountType volCl2 = cl2Reg.volume();

    // Save the indices for the reactants because they will be used for every partials
    indices(0) = _reactants[0];
    for (auto i : speciesRangeNoI) {
    	indices(i() + 1) = _reactantMomentIds[0][i()];
    }
    indices(numSpeciesNoI + 2) = _reactants[1];
    for (auto i : speciesRangeNoI) {
    	indices(numSpeciesNoI + i() + 3) = _reactantMomentIds[1][i()];
    }
    
    // Compute the partials for the 0th order moments
    // Compute the values (d / dL_0^A)
    double temp = _coefs(0, 0, 0, 0) * concentrations[_reactants[1]];
    for (auto i : speciesRangeNoI) {
    	temp += _coefs(0, i() + 1, 0, 0) *
                concentrations[_reactantMomentIds[1][i()]];
    }
    // First for the first reactant
    indices(2 * (numSpeciesNoI + 1) + 1) = _reactants[0];
    values(0) = - _rate(gridIndex) * temp / (double) volCl1;
    // Second reactant
    indices(2 * (numSpeciesNoI + 1) + 2) = _reactants[1];
    values(2 * (numSpeciesNoI + 1) + 1) = - _rate(gridIndex) * temp / (double) volCl2;
    // For the products
    for (std::size_t p : {0,1}) {
        auto prodId = _products[p];
        if (prodId == invalid) {
            continue;
        }
        auto prod = _network->getCluster(prodId);
        const auto& prodReg = prod.getRegion();
        AmountType volProd = prodReg.volume();
        indices(2 * (numSpeciesNoI + 1) + 3 + p) = _products[p];
        values(2 * (p + 2) * (numSpeciesNoI + 1) + 1) = _rate(gridIndex) * temp / (double) volProd;
    }
    
    // (d / dL_1^A)
    for (auto i : speciesRangeNoI) {
    	temp = _coefs(i() + 1, 0, 0, 0) * concentrations[_reactants[1]];
        for (auto j : speciesRangeNoI) {
        	temp += _coefs(i() + 1, j() + 1, 0, 0) * concentrations[_reactantMomentIds[1][j()]];
        }
        // First reactant
        values(i() + 1) = - _rate(gridIndex) * temp / (double) volCl1;
        // second reactant
        values(2 * (numSpeciesNoI + 1) + i() + 2) = - _rate(gridIndex) * temp / (double) volCl2;
        // For the products
        for (std::size_t p : {0,1}) {
            auto prodId = _products[p];
            if (prodId == invalid) {
                continue;
            }
            auto prod = _network->getCluster(prodId);
            const auto& prodReg = prod.getRegion();
            AmountType volProd = prodReg.volume();
            values(2 * (p + 2) * (numSpeciesNoI + 1) + i() + 2) = _rate(gridIndex) * temp / (double) volProd;
        }
    }
    
    // (d / dL_0^B)
    temp = _coefs(0, 0, 0, 0) * concentrations[_reactants[0]];
    for (auto i : speciesRangeNoI) {
    	temp += _coefs(i() + 1, 0, 0, 0) *
                concentrations[_reactantMomentIds[0][i()]];
    }
    values(numSpeciesNoI + 2) = - _rate(gridIndex) * temp / (double) volCl1;
    values(3 * (numSpeciesNoI + 1) + 1) = - _rate(gridIndex) * temp / (double) volCl2;
    for (std::size_t p : {0,1}) {
        auto prodId = _products[p];
        if (prodId == invalid) {
            continue;
        }
        auto prod = _network->getCluster(prodId);
        const auto& prodReg = prod.getRegion();
        AmountType volProd = prodReg.volume();
        values((2 * (p + 2) + 1) * (numSpeciesNoI + 1) + 1) = _rate(gridIndex) * temp / (double) volProd;
    }
    
    // (d / dL_1^B)
    for (auto i : speciesRangeNoI) {
    	temp = _coefs(0, i() + 1, 0, 0) * concentrations[_reactants[0]];
        for (auto j : speciesRangeNoI) {
        	temp += _coefs(j() + 1, i() + 1, 0, 0) * concentrations[_reactantMomentIds[0][j()]];
        }
        values(numSpeciesNoI + i() + 3) = - _rate(gridIndex) * temp / (double) volCl1;
        values(3 * (numSpeciesNoI + 1) + i() + 2) = - _rate(gridIndex) * temp / (double) volCl2;
        for (std::size_t p : {0,1}) {
            auto prodId = _products[p];
            if (prodId == invalid) {
                continue;
            }
            auto prod = _network->getCluster(prodId);
            const auto& prodReg = prod.getRegion();
            AmountType volProd = prodReg.volume();
            values((2 * (p + 2) + 1) * (numSpeciesNoI + 1) + i() + 2) = _rate(gridIndex) * temp / (double) volProd;
        }
    }
        
    // Take care of the first moments
    for (auto k : speciesRangeNoI) {
        // First for the first reactant
        indices(2 * (numSpeciesNoI + 1) + nProd + 3 + k()) = _reactantMomentIds[0][k()];
        // (d / dL_0^A)
        temp = _coefs(0, 0, 0, k() + 1) * concentrations[_reactants[1]];
        for (auto j : speciesRangeNoI) {
            temp += _coefs(0, j() + 1, 0, k() + 1) * concentrations[_reactantMomentIds[1][j()]];
        }
        values(2 * (nProd + 3) * (numSpeciesNoI + 1) + k() + 1) = - _rate(gridIndex) * temp / (double) volCl1;
        // (d / dL_1^A)
        for (auto i : speciesRangeNoI) {
            temp = _coefs(i() + 1, 0, 0, k() + 1) * concentrations[_reactants[1]];
            for (auto j : speciesRangeNoI) {
                temp += _coefs(i() + 1, j() + 1, 0, k() + 1) * concentrations[_reactantMomentIds[1][j()]];
            }
            values((2 * (nProd + 3) + 1) * (numSpeciesNoI + 1) + k() * numSpeciesNoI + i()) 
            = - _rate(gridIndex) * temp / (double) volCl1;
        }
        // (d / dL_0^B)
        temp = _coefs(0, 0, 0, k() + 1) * concentrations[_reactants[0]];
        for (auto j : speciesRangeNoI) {
            temp += _coefs(j() + 1, 0, 0, k() + 1) * concentrations[_reactantMomentIds[0][j()]];
        }
        values((2 * (nProd + 3) + 1) * (numSpeciesNoI + 1) + k() + numSpeciesNoI * numSpeciesNoI) 
        = - _rate(gridIndex) * temp / (double) volCl1;
        // (d / dL_1^B)
        for (auto i : speciesRangeNoI) {
            temp = _coefs(0, i() + 1, 0, k() + 1) * concentrations[_reactants[0]];
            for (auto j : speciesRangeNoI) {
                temp += _coefs(j() + 1, i() + 1, 0, k() + 1) * concentrations[_reactantMomentIds[0][j()]];
            }
            values(2 * (nProd + 4) * (numSpeciesNoI + 1) + numSpeciesNoI * numSpeciesNoI + k() * numSpeciesNoI + i()) 
            = - _rate(gridIndex) * temp / (double) volCl1;
        }
    }

    // Take care of the first moments
    for (auto k : speciesRangeNoI) {
        // First for the second reactant
        indices(2 * (numSpeciesNoI + 1) + nProd + numSpeciesNoI + 3 + k()) = _reactantMomentIds[1][k()];
        // (d / dL_0^A)
        temp = _coefs(0, 0, 1, k() + 1) * concentrations[_reactants[1]];
        for (auto j : speciesRangeNoI) {
            temp += _coefs(0, j() + 1, 1, k() + 1) * concentrations[_reactantMomentIds[1][j()]];
        }
        values(2 * (nProd + 4) * (numSpeciesNoI + 1) + 2 * numSpeciesNoI * numSpeciesNoI + k()) 
        = - _rate(gridIndex) * temp / (double) volCl2;
        // (d / dL_1^A)
        for (auto i : speciesRangeNoI) {
            temp = _coefs(i() + 1, 0, 1, k() + 1) * concentrations[_reactants[1]];
            for (auto j : speciesRangeNoI) {
                temp += _coefs(i() + 1, j() + 1, 1, k() + 1) * concentrations[_reactantMomentIds[1][j()]];
            }
            values((2 * (nProd + 4) + 1) * (numSpeciesNoI + 1) + 2 * numSpeciesNoI * numSpeciesNoI + k() * numSpeciesNoI + i()) 
            = - _rate(gridIndex) * temp / (double) volCl2;
        }
        // (d / dL_0^B)
        temp = _coefs(0, 0, 1, k() + 1) * concentrations[_reactants[0]];
        for (auto j : speciesRangeNoI) {
            temp += _coefs(j() + 1, 0, 1, k() + 1) * concentrations[_reactantMomentIds[0][j()]];
        }
        values((2 * (nProd + 4) + 1) * (numSpeciesNoI + 1) + 3 * numSpeciesNoI * numSpeciesNoI + k()) 
        = - _rate(gridIndex) * temp / (double) volCl2;
        // (d / dL_1^B)
        for (auto i : speciesRangeNoI) {
            temp = _coefs(0, i() + 1, 1, k() + 1) * concentrations[_reactants[0]];
            for (auto j : speciesRangeNoI) {
                temp += _coefs(j() + 1, i() + 1, 1, k() + 1) * concentrations[_reactantMomentIds[0][j()]];
            }
            values(2 * (nProd + 5) * (numSpeciesNoI + 1) + 3 * numSpeciesNoI * numSpeciesNoI + k() * numSpeciesNoI + i()) 
            = - _rate(gridIndex) * temp / (double) volCl2;
        }
    }
    
    // Loop on the products
    for (std::size_t p : {0,1}) {
        auto prodId = _products[p];
        if (prodId == invalid) {
            continue;
        }

        auto prod = _network->getCluster(prodId);
        const auto& prodReg = prod.getRegion();
        AmountType volProd = prodReg.volume();

        // Take care of the first moments
        for (auto k : speciesRangeNoI) {
            indices(2 * (numSpeciesNoI + 1) + nProd + 3 + (p + 2) * numSpeciesNoI + k()) = _productMomentIds[p][k()];
            // (d / dL_0^A)
            temp = _coefs(0, 0, p + 2, k() + 1) * concentrations[_reactants[1]];
            for (auto j : speciesRangeNoI) {
                temp += _coefs(0, j() + 1, p + 2, k() + 1) * concentrations[_reactantMomentIds[1][j()]];
            }
            values(2 * (nProd + 5 + p) * (numSpeciesNoI + 1) + 2 * (p + 2) * numSpeciesNoI * numSpeciesNoI + k()) 
            = _rate(gridIndex) * temp / (double) volProd;
            // (d / dL_1^A)
            for (auto i : speciesRangeNoI) {
                temp = _coefs(i() + 1, 0, p + 2, k() + 1) * concentrations[_reactants[1]];
                for (auto j : speciesRangeNoI) {
                    temp += _coefs(i() + 1, j() + 1, p + 2, k() + 1) * concentrations[_reactantMomentIds[1][j()]];
                }
                values((2 * (nProd + 5 + p) + 1) * (numSpeciesNoI + 1) + 2 * (p + 2) * numSpeciesNoI * numSpeciesNoI 
                		+ k() * numSpeciesNoI + i()) = _rate(gridIndex) * temp / (double) volProd;
            }
            // (d / dL_0^B)
            temp = _coefs(0, 0, p + 2, k() + 1) * concentrations[_reactants[0]];
            for (auto j : speciesRangeNoI) {
                temp += _coefs(j() + 1, 0, p + 2, k() + 1) * concentrations[_reactantMomentIds[0][j()]];
            }
            values((2 * (nProd + 5 + p) + 1) * (numSpeciesNoI + 1) + (2 * (p + 2) + 1) * numSpeciesNoI * numSpeciesNoI 
            		+ k()) = _rate(gridIndex) * temp / (double) volProd;
            // (d / dL_1^B)
            for (auto i : speciesRangeNoI) {
                temp = _coefs(0, i() + 1, p + 2, k() + 1) * concentrations[_reactants[0]];
                for (auto j : speciesRangeNoI) {
                    temp += _coefs(j() + 1, i() + 1, p + 2, k() + 1) * concentrations[_reactantMomentIds[0][j()]];
                }
                values(2 * (nProd + 6 + p) * (numSpeciesNoI + 1) + (2 * (p + 2) + 1) * numSpeciesNoI * numSpeciesNoI 
                		+ k() * numSpeciesNoI + i()) = _rate(gridIndex) * temp / (double) volProd;
            }
        }
    }
}

template <typename TImpl>
template <typename TDerived>
inline void
ReactionNetwork<TImpl>::Reaction<TDerived>::dissociationPartialDerivatives(
    ConcentrationsView concentrations, Kokkos::View<std::size_t*> indices,
    Kokkos::View<double*> values, std::size_t gridIndex)
{
    using AmountType = typename NetworkType::AmountType;
    constexpr auto numSpeciesNoI = NetworkType::getNumberOfSpeciesNoI();
    constexpr auto speciesRangeNoI = NetworkType::getSpeciesRangeNoI();

    // Compute the total number of elements in each cluster
    auto cl = _network->getCluster(_reactants[0]);
    const auto& clReg = cl.getRegion();
    AmountType volCl = clReg.volume();
    auto prod1 = _network->getCluster(_products[0]);
    const auto& prod1Reg = prod1.getRegion();
    AmountType volProd1 = prod1Reg.volume();
    auto prod2 = _network->getCluster(_products[1]);
    const auto& prod2Reg = prod2.getRegion();
    AmountType volProd2 = prod2Reg.volume();

    // Save the indices for the reactant because they will be used for every partials
    indices(0) = _reactants[0];
    for (auto i : speciesRangeNoI) {
    	indices(i() + 1) = _reactantMomentIds[0][i()];
    }
    
    // Compute the partials for the 0th order moments
    // First for the reactant
    double df = - _rate(gridIndex) / (double) volCl;
    // Keep its index
    indices(numSpeciesNoI + 2) = _reactants[0];
    // Compute the values
    values(0) = df * _coefs(0, 0, 0, 0);
    for (auto i : speciesRangeNoI) {
    	values(i() + 1) = df * _coefs(i() + 1, 0, 0, 0);
    }
    // For the first product
    df = _rate(gridIndex) / (double) volProd1;
    indices(numSpeciesNoI + 3) = _products[0];
    values(numSpeciesNoI + 2) = df * _coefs(0, 0, 0, 0);
    for (auto i : speciesRangeNoI) {
    	values(numSpeciesNoI + i() + 3) = df * _coefs(i() + 1, 0, 0, 0);
    }
    // For the second product
    df = _rate(gridIndex) / (double) volProd2;
    indices(numSpeciesNoI + 4) = _products[1];
    values(2 * (numSpeciesNoI + 1) + 1) = df * _coefs(0, 0, 0, 0);
    for (auto i : speciesRangeNoI) {
    	values(2 * (numSpeciesNoI + 1) + i() + 2) = df * _coefs(i() + 1, 0, 0, 0);
    }
    
    // Take care of the first moments
    for (auto k : speciesRangeNoI) {
        // First for the reactant
        // TODO compute the prefactor related to the dispersion, it can be
        // moved to the coefs maybe
        double prefactor = 1.0;
        df = - _rate(gridIndex) * prefactor / (double) volCl;
        // Keep its index
        indices(numSpeciesNoI + (3 * k()) + 5) = _reactantMomentIds[0][k()];
        // Compute the values
        values((3 * k() + 3) * (numSpeciesNoI + 1) + 1) = df * _coefs(0, 0, 0, k() + 1);
        for (auto i : speciesRangeNoI) {
        	values((3 * k() + 3) * (numSpeciesNoI + 1) + i() + 2) = df * _coefs(i() + 1, 0, 0, k() + 1);
        }
        // For the first product
        prefactor = 1.0;
        df = _rate(gridIndex) * prefactor / (double) volProd1;
        indices(numSpeciesNoI + (3 * k()) + 6) = _productMomentIds[0][k()];
        values((3 * k() + 4) * (numSpeciesNoI + 1) + 1) = df * _coefs(0, 0, 1, k() + 1);
        for (auto i : speciesRangeNoI) {
        	values((3 * k() + 4) * (numSpeciesNoI + 1) + i() + 2) = df * _coefs(i() + 1, 0, 1, k() + 1);
        }
        // For the second product
        prefactor = 1.0;
        df = _rate(gridIndex) * prefactor / (double) volProd2;
        indices(numSpeciesNoI + (3 * k()) + 7) = _productMomentIds[1][k()];
        values((3 * k() + 5) * (numSpeciesNoI + 1) + 1) = df * _coefs(0, 0, 2, k() + 1);
        for (auto i : speciesRangeNoI) {
        	values((3 * k() + 5) * (numSpeciesNoI + 1) + i() + 2) = df * _coefs(i() + 1, 0, 2, k() + 1);
        }
    }
}
}
}
