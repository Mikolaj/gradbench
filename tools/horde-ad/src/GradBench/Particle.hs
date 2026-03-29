{-# LANGUAGE OverloadedLists, OverloadedStrings #-}
module GradBench.Particle (Input, Output, rr, ff, fr, rf) where

import Data.Aeson ((.:))
import Data.Aeson qualified as JSON
import GradBench.GD
import HordeAd
import HordeAd.Core.Adaptor

data Input = Input Double

type Output = Double

instance JSON.FromJSON Input where
  parseJSON = JSON.withObject "input" $ \o ->
    Input <$> o .: "w"

magnitude_squaredK :: (NumScalar a, ADReady target)
                   => target (TKScalar a) -> target (TKScalar a)
{-# INLINE magnitude_squaredK #-}
magnitude_squaredK t' = tlet t' $ \t -> t * t

scaleK :: (NumScalar a, ADReady target)
       => a -> target (TKScalar a) -> target (TKScalar a)
{-# INLINE scaleK #-}
scaleK x v = kconcrete x * v

cgrad2_fwdK
  :: forall src r tgt target.
     ( src ~ ADVal target (TKScalar r)
     , NumScalar r, ADTensorScalar r ~ r
     , tgt ~ ADVal target (TKScalar r)
     , ADReadyNoLet target, ShareTensor target )
  => (src -> tgt)  -- ^ the objective function
  -> DValue src
  -> ( target (TKScalar r)
     , DValue src )  -- morally DValue (ADTensorKind src)
{-# INLINE cgrad2_fwdK #-}
cgrad2_fwdK f x = cjvp2 f x 1

cgrad_fwdK2
  :: forall src r tgt target.
     ( src ~ (ADVal target (TKScalar r), ADVal target (TKScalar r))
     , NumScalar r, ADTensorScalar r ~ r
     , tgt ~ ADVal target (TKScalar r)
     , ADReadyNoLet target, ShareTensor target
     , ShareTensor (PrimalOf target), ShareTensor (PlainOf target) )
  => (src -> tgt)  -- ^ the objective function
  -> DValue src
  -> DValue src  -- morally DValue (ADTensorKind src)
{-# INLINE cgrad_fwdK2 #-}
cgrad_fwdK2 f x = (cjvp f x (1, 0), cjvp f x (0, 1))

type Point a = (a, a)

pplus :: (Num a) => Point a -> Point a -> Point a
pplus u v = (fst u + fst v, snd u + snd v)

ktimesp :: (Num a) => a -> Point a -> Point a
ktimesp k u = (k * fst u, k * snd u)

-- No sharing, so not good for the symbolic pipeline.
sqr :: (Floating a) => a -> a
sqr x = x * x

dist :: (Floating a) => Point a -> Point a -> a
dist u v = sqrt (sqr (fst u - fst v) + sqr (snd u - snd v))

accel :: (Floating a) => [Point a] -> Point a -> a
accel charges x = sum $ map (\p -> recip (dist p x)) charges

naiveEuler
  :: (ADReadyNoLet target, Ord (target (TKScalar Double)))
  => ([Point (target (TKScalar Double))] -> Point (target (TKScalar Double))
      -> Point (target (TKScalar Double)))
  -> target (TKScalar Double)
  -> target (TKScalar Double)
{-# INLINE naiveEuler #-}
naiveEuler accel' w =
  let x_initial = (0, 8)
      xdot_initial = (0.75, 0)
      (x, xdot) = loop x_initial xdot_initial
      delta_t_f = - (snd x) / snd xdot
      x_t_f = x `pplus` (delta_t_f `ktimesp` xdot)
  in sqr (fst x_t_f)
 where
  charges = [(10, 10 - w), (10, 0)]
  delta_t = 1e-1
  loop x xdot =
    let xddot = (-1) `ktimesp` accel' charges x
        x_new = x `pplus` (delta_t `ktimesp` xdot)
    in if snd x_new > 0
       then loop x_new $ xdot `pplus` (delta_t `ktimesp` xddot)
       else (x, xdot)

-- TODO: this is very slow; see the comment in Saddle.hs
rr, ff, fr, rf :: Input -> Output
rr (Input w0) =
  unConcrete
  $ multivariateArgmin magnitude_squaredK scaleK g (kconcrete w0)
 where
  accel' :: ( ADReadyNoLet target, ShareTensor target
            , ShareTensor (PrimalOf target), ShareTensor (PlainOf target) )
         => [Point (target (TKScalar Double))]
         -> Point (target (TKScalar Double))
         -> Point (target (TKScalar Double))
  accel' charges = cgrad (accel $ map (\(x, y) ->
                            (kfromPrimal x, kfromPrimal y)) charges)
  f :: ( ADReadyNoLet target, ShareTensor target
       , ShareTensor (PrimalOf target), ShareTensor (PlainOf target)
       , Ord (target (TKScalar Double)) )
    => target (TKScalar Double) -> target (TKScalar Double)
  f w = naiveEuler accel' w
  g :: ( ADReadyNoLet target, ShareTensor target
       , ShareTensor (PrimalOf target), ShareTensor (PlainOf target)
       , Ord (target (TKScalar Double)) )
    => target (TKScalar Double)
    -> (target (TKScalar Double), target (TKScalar Double))
  g a = cgrad2 f a
ff (Input w0) =
  unConcrete
  $ multivariateArgmin magnitude_squaredK scaleK g (kconcrete w0)
 where
  accel' :: ( ADReadyNoLet target, ShareTensor target
            , ShareTensor (PrimalOf target), ShareTensor (PlainOf target) )
         => [Point (target (TKScalar Double))]
         -> Point (target (TKScalar Double))
         -> Point (target (TKScalar Double))
  accel' charges = cgrad_fwdK2 (accel $ map (\(x, y) ->
                                  (kfromPrimal x, kfromPrimal y)) charges)
  f :: ( ADReadyNoLet target, ShareTensor target
       , ShareTensor (PrimalOf target), ShareTensor (PlainOf target)
       , Ord (target (TKScalar Double)) )
    => target (TKScalar Double) -> target (TKScalar Double)
  f w = naiveEuler accel' w
  g :: ( ADReadyNoLet target, ShareTensor target
       , ShareTensor (PrimalOf target), ShareTensor (PlainOf target)
       , Ord (target (TKScalar Double)) )
    => target (TKScalar Double)
    -> (target (TKScalar Double), target (TKScalar Double))
  g a = cgrad2_fwdK f a
fr (Input w0) =
  unConcrete
  $ multivariateArgmin magnitude_squaredK scaleK g (kconcrete w0)
 where
  accel' :: ( ADReadyNoLet target, ShareTensor target
            , ShareTensor (PrimalOf target), ShareTensor (PlainOf target) )
         => [Point (target (TKScalar Double))]
         -> Point (target (TKScalar Double))
         -> Point (target (TKScalar Double))
  accel' charges = cgrad (accel $ map (\(x, y) ->
                            (kfromPrimal x, kfromPrimal y)) charges)
  f :: ( ADReadyNoLet target, ShareTensor target
       , ShareTensor (PrimalOf target), ShareTensor (PlainOf target)
       , Ord (target (TKScalar Double)) )
    => target (TKScalar Double) -> target (TKScalar Double)
  f w = naiveEuler accel' w
  g :: ( ADReadyNoLet target, ShareTensor target
       , ShareTensor (PrimalOf target), ShareTensor (PlainOf target)
       , Ord (target (TKScalar Double)) )
    => target (TKScalar Double)
    -> (target (TKScalar Double), target (TKScalar Double))
  g a = cgrad2_fwdK f a
rf (Input w0) =
  unConcrete
  $ multivariateArgmin magnitude_squaredK scaleK g (kconcrete w0)
 where
  accel' :: ( ADReadyNoLet target, ShareTensor target
            , ShareTensor (PrimalOf target), ShareTensor (PlainOf target) )
         => [Point (target (TKScalar Double))]
         -> Point (target (TKScalar Double))
         -> Point (target (TKScalar Double))
  accel' charges = cgrad_fwdK2 (accel $ map (\(x, y) ->
                                  (kfromPrimal x, kfromPrimal y)) charges)
  f :: ( ADReadyNoLet target, ShareTensor target
       , ShareTensor (PrimalOf target), ShareTensor (PlainOf target)
       , Ord (target (TKScalar Double)) )
    => target (TKScalar Double) -> target (TKScalar Double)
  f w = naiveEuler accel' w
  g :: ( ADReadyNoLet target, ShareTensor target
       , ShareTensor (PrimalOf target), ShareTensor (PlainOf target)
       , Ord (target (TKScalar Double)) )
    => target (TKScalar Double)
    -> (target (TKScalar Double), target (TKScalar Double))
  g a = cgrad2 f a
