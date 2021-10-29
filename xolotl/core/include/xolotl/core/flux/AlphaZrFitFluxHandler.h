#ifndef ALPHAZRFITFLUXHANDLER_H
#define ALPHAZRFITFLUXHANDLER_H

#include <cmath>

#include <xolotl/core/flux/FluxHandler.h>
#include <xolotl/core/network/ZrReactionNetwork.h>

namespace xolotl
{
namespace core
{
namespace flux
{
/**
 * This class realizes the IFluxHandler interface to calculate the incident
 * fluxes for an alpha Zr material.
 */
class AlphaZrFitFluxHandler : public FluxHandler
{
private:
	/**
	 * \see FluxHandler.h
	 */
	double
	FitFunction(double x)
	{
		// Not actually used
		return 0.0;
	}

public:
	/**
	 * The constructor
	 */
	AlphaZrFitFluxHandler(const options::IOptions& options) :
		FluxHandler(options)
	{
	}

	/**
	 * The Destructor
	 */
	~AlphaZrFitFluxHandler()
	{
	}

	/**
	 * \see IFluxHandler.h
	 */
    
    // Set the maximum cluster sizes to set a generation flux to
    int maxSizeI = 70;
    int maxSizeV = 70;
    
	void
	initializeFluxHandler(network::IReactionNetwork& network, int surfacePos,
		std::vector<double> grid)
	{
		// Only defined in 0D
		if (xGrid.size() == 0) {
			// Add an empty vector
			std::vector<double> tempVector;
			incidentFluxVec.push_back(tempVector);
		}

		using NetworkType = network::ZrReactionNetwork;
		auto zrNetwork = dynamic_cast<NetworkType*>(&network);
        int numMobileI = 0;
        int numMobileV = 0;
        if (maxSizeI < 10) numMobileI = maxSizeI+1;
        else numMobileI = 10;
        if (maxSizeV < 10) numMobileV = maxSizeV+1;
        else numMobileV = 10;

		// Set the flux index corresponding the mobile interstitial clusters (n < 10)
        NetworkType::Composition comp = NetworkType::Composition::zero();
        comp[NetworkType::Species::I] = 1;
        std::ostringstream oss;
        auto cluster = zrNetwork->findCluster(comp, plsm::onHost);
        for (int i = 1; i < numMobileI; i++){
            comp[NetworkType::Species::I] = i;
            cluster = zrNetwork->findCluster(comp, plsm::onHost);
            if (cluster.getId() == NetworkType::invalidIndex()) {
                throw std::runtime_error("\nThe current interstitial cluster is not "
                                         "present in the network, "
                                         "cannot use the flux option!");
            }
            fluxIndices.push_back(cluster.getId());
        }
        
        
        if (maxSizeI > 9){
            // Set the flux index corresponding to the immobile Prismatic interstitial clusters (n < 10)
            for (int i = 10; i < (maxSizeI+1); i++){
                comp[NetworkType::Species::I] = i;
                cluster = zrNetwork->findCluster(comp, plsm::onHost);
                if (cluster.getId() == NetworkType::invalidIndex()) {
                    throw std::runtime_error("\nThe current interstitial cluster is not "
                                             "present in the network, "
                                             "cannot use the flux option!");
                }
                fluxIndices.push_back(cluster.getId());
            }
        }
        

        // Set the flux index corresponding the mobile vacancy clusters (n < 10)
		comp[NetworkType::Species::I] = 0;
        for (int i = 1; i < numMobileV; i++){
            comp[NetworkType::Species::V] = i;
            cluster = zrNetwork->findCluster(comp, plsm::onHost);
            if (cluster.getId() == NetworkType::invalidIndex()) {
                throw std::runtime_error("\nThe current vacancy cluster is not "
                                         "present in the network, "
                                         "cannot use the flux option!");
            }
            fluxIndices.push_back(cluster.getId());
        }
        
        
        if (maxSizeV > 9){
         
            // Set the flux index corresponding to the immobile Prismatic vacancy clusters (n < 10)
            for (int i = 10; i < (maxSizeV+1); i++){
                comp[NetworkType::Species::V] = i;
                cluster = zrNetwork->findCluster(comp, plsm::onHost);
                if (cluster.getId() == NetworkType::invalidIndex()) {
                    throw std::runtime_error("\nThe current interstitial cluster is not "
                                             "present in the network, "
                                             "cannot use the flux option!");
                }
                fluxIndices.push_back(cluster.getId());
            }
            
        }
        

		return;
	}
         
	/**
	 * \see IFluxHandler.h
	 */
	void
	computeIncidentFlux(
		double currentTime, double* updatedConcOffset, int xi, int surfacePos)
	{
		// Define only for a 0D case
		if (incidentFluxVec[0].size() == 0) {
            
            // Define the range of possible cluster sizes, and their respective fluxes in nm^-3 s^-1
            // Multiply by 1.0e+21 to get to cm^-3 s^-1 if you want to check values
            //int clusterSize[9] = {1, 2, 3, 4, 5, 10, 20, 40, 70};
            //float fluxI[9] = {2.3594721220018477e-07, 2.166655565558331e-08, 6.570120819263884e-09, 3.894982908643517e-09, 2.0534450133333332e-09, 4.9103974466666655e-09, 1.8990249399999996e-09, 5.712584999999999e-10, 2.40225e-11};
            //float fluxV[9] = {1.9162900023661088e-07, 2.091085499806479e-08, 7.352860982986093e-09, 3.5985803572870338e-09, 2.144962155310184e-09, 6.174987030620367e-09, 3.61472944e-09, 1.45843282e-09, 3.8405969999999995e-10};
            float fluxI[70] = {2.3591362027648328e-07, 2.1695247321410436e-08, 6.603086798728652e-09, 3.968748769614071e-09, 2.1030674767333324e-09, 1.7878681529999996e-09, 1.5171312549999996e-09, 5.484931699999999e-10, 4.082159399999999e-10, 6.684360369999999e-10, 4.3549699500000005e-10, 1.50942693e-10, 2.0196496799999991e-10, 1.42698872e-10, 3.386073059999999e-10, 1.65978146e-10, 9.845374200000001e-11, 6.192678299999998e-11, 1.7366884200000002e-10, 1.1389926199999999e-10, 7.1229639e-11, 1.9257160000000002e-11, 1.3854356899999995e-10, 4.515648899999999e-11, 2.8885739999999996e-11, 1.1123857499999998e-10, 0, 3.5527909e-11, 9.628580000000001e-12, 3.5527909e-11, 9.628580000000001e-12, 0, 2.5899328999999996e-11, 2.5899328999999996e-11, 0, 2.5899328999999996e-11, 0, 9.628580000000001e-12, 0, 0, 9.628580000000001e-12, 0, 2.5899328999999996e-11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            float fluxV[70] = {1.7811390508204618e-07, 1.6930667065081818e-08, 7.0735654428151215e-09, 3.3754591361812474e-09, 1.840000680680739e-09, 1.6146127259999994e-09, 2.1627167789999997e-09, 1.015531922447914e-09, 6.790095549999998e-10, 6.365080860000001e-10, 5.00133545e-10, 6.908325919999998e-10, 6.111473429999999e-10, 3.894946049999999e-10, 2.2615779300000004e-10, 4.01893729e-10, 1.45108826e-10, 1.6787280799999993e-10, 2.0438660999999995e-10, 1.6719651500000004e-10, 1.4427961800000002e-10, 1.61956032e-10, 9.429843900000001e-11, 7.504127899999999e-11, 4.2669623e-11, 1.01114429e-10, 1.5158115299999994e-10, 3.5527909e-11, 3.3041043e-11, 1.2437397699999997e-10, 0, 7.1229639e-11, 6.856895199999998e-11, 4.515648899999999e-11, 6.731392999999998e-11, 3.3041043e-11, 0, 8.501352199999999e-11, 2.8885739999999996e-11, 0, 3.5527909e-11, 0, 5.1972479000000005e-11, 0, 9.628580000000001e-12, 0, 9.628580000000001e-12, 1.9257160000000002e-11, 1.9257160000000002e-11, 3.3041043e-11, 9.628580000000001e-12, 1.9257160000000002e-11, 0, 0, 0, 2.5899328999999996e-11, 2.5899328999999996e-11, 3.428447599999999e-11, 9.628580000000001e-12, 0, 0, 0, 2.5899328999999996e-11, 0, 0, 1.9257160000000002e-11, 0, 0, 0, 0};
           
            // Set the interstitial fluxes
            for (int i = 0; i < maxSizeI; i++){
                updatedConcOffset[fluxIndices[i]] += fluxI[i];
            }

            // Set the vacancy fluxes
            for (int i = 0; i < maxSizeV; i++){
                updatedConcOffset[fluxIndices[i+maxSizeI]] += fluxV[i];

            }
            
		}

		else {
			throw std::runtime_error(
				"\nThe alpha Zr problem is not defined for more than 0D!");
		}

		return;
	}
};
// end class AlphaZrFitFluxHandler

} // namespace flux
} // namespace core
} // namespace xolotl

#endif
