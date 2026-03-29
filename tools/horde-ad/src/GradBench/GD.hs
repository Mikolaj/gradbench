-- | This is not an eval, but rather a (simple) implementation of
-- Gradient Descent. It is very naive, but this is
-- sufficient for the 'particle' and 'saddle' evals.
module GradBench.GD
  ( magnitude_squaredR, scaleR
  , multivariateArgmin, multivariateArgmax, multivariateMax
  )
where

import HordeAd

magnitude_squaredR :: (NumScalar a, ADReady target)
                   => target (TKR 1 a) -> target (TKScalar a)
{-# INLINE magnitude_squaredR #-}
magnitude_squaredR t' = tlet t' $ \t -> rdot0 t t

scaleR :: (NumScalar a, ADReady target)
       => a -> target (TKR 1 a) -> target (TKR 1 a)
{-# INLINE scaleR #-}
scaleR x v = rrepl (rshape v) x * v

-- | The solver must be invoked with a function returning a pair: the cost
-- and the gradient.
multivariateArgmin
  :: forall a y target.
     ( NumScalar a, Differentiable a, ADReady target, Num (target y)
     , Ord (target (TKScalar a)) )
  => (target y -> target (TKScalar a))
  -> (a -> target y -> target y)
  -> (target y -> (target (TKScalar a), target y))
  -> target y -> target y
{-# INLINE multivariateArgmin #-}
multivariateArgmin magnitude_squared scale fg x0 = loop (x0, fx0, gx0, 1e-5, 0)
 where
  magnitude :: target y -> target (TKScalar a)
  magnitude = sqrt . magnitude_squared
  distance_squared :: target y -> target y -> target (TKScalar a)
  distance_squared u v = magnitude_squared (u - v)
  distance :: target y -> target y -> target (TKScalar a)
  distance u v = sqrt $ distance_squared u v
  (fx0, gx0) = fg x0
  loop :: (target y, target (TKScalar a), target y, a, Int) -> target y
  loop (x, fx, gx, eta, i)
    | magnitude gx <= 1e-5 = x
    | i == 10 = loop (x, fx, gx, 2 * eta, 0)
    | distance x x_prime <= 1e-5 = x
    | fx_prime < fx = loop (x_prime, fx_prime, gx_prime, eta, i + 1)
    | otherwise = loop (x, fx, gx, eta / 2, 0)
   where
     x_prime = x - (eta `scale` gx)
     (fx_prime, gx_prime) = fg x_prime

multivariateArgmax
  :: ( NumScalar a, Differentiable a, ADReady target, Num (target y)
     , Ord (target (TKScalar a)) )
  => (target y -> target (TKScalar a))
  -> (a -> target y -> target y)
  -> (target y -> (target (TKScalar a), target y))
  -> target y -> target y
{-# INLINE multivariateArgmax #-}
multivariateArgmax magnitude_squared scale fg =
  multivariateArgmin magnitude_squared scale (\arg -> let (c, g) = fg arg
                                                      in (-c, -g))

multivariateMax
  :: ( NumScalar a, Differentiable a, ADReady target, Num (target y)
     , Ord (target (TKScalar a)) )
  => (target y -> target (TKScalar a))
  -> (a -> target y -> target y)
  -> (target y -> (target (TKScalar a), target y))
  -> target y -> target (TKScalar a)
{-# INLINE multivariateMax #-}
multivariateMax magnitude_squared scale fg x =
  fst $ fg $ multivariateArgmax magnitude_squared scale fg x
