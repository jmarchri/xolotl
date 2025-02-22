#pragma once

#include <memory>

#include <xolotl/perf/ITimer.h>

namespace xolotl
{
namespace perf
{
/**
 * A class for managing timer start/stop lifetime by code scope.
 * Used to simplify a common use case for a timer (starting timer when
 * enter a scope, and stopping timer when leave the scope regardless
 * of how we leave the scope).
 */
struct ScopedTimer
{
	/// The timer that should be active in the struct's scope.
	std::shared_ptr<ITimer> timer;

	ScopedTimer(std::shared_ptr<ITimer> _timer) : timer(_timer)
	{
		timer->start();
	}

	~ScopedTimer(void)
	{
		timer->stop();
	}
};
} // end namespace perf
} // end namespace xolotl
