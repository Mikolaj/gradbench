{-# LANGUAGE OverloadedLists, OverloadedStrings #-}
module GradBench.Saddle (Input, Output, rr, ff, fr, rf) where

import Data.Aeson ((.:))
import Data.Aeson qualified as JSON
import Data.Array.Nested qualified as Nested
import Data.Array.Nested.Lemmas
import Data.Array.Nested.Shaped.Shape
import Data.Type.Equality ((:~:) (Refl))
import Data.Vector.Storable qualified as VS
import GradBench.GD
import HordeAd
import HordeAd.Core.Adaptor

data Input = Input (Double, Double)

type Output = VS.Vector Double

instance JSON.FromJSON Input where
  parseJSON = JSON.withObject "input" $ \o ->
    Input <$> o .: "start"

magnitude_squaredS :: (KnownShS sh, NumScalar a, ADReady target)
                   => target (TKS sh a) -> target (TKScalar a)
{-# INLINE magnitude_squaredS #-}
magnitude_squaredS t' = tlet t' $ \t -> sdot0 t t

scaleS :: (NumScalar a, ADReady target)
       => a -> target (TKS '[2] a) -> target (TKS '[2] a)
{-# INLINE scaleS #-}
scaleS x v = sconcrete (Nested.sreplicatePrim (SNat @2 :$$ ZSS) x) * v

cgrad2_fwdS
  :: forall src r tgt target sh.
     ( src ~ ADVal target (TKS sh r)
     , NumScalar r, ADTensorScalar r ~ r, KnownShS sh
     , tgt ~ ADVal target (TKScalar r)
     , ADReadyNoLet target, ShareTensor target
     , ShareTensor (PrimalOf target), ShareTensor (PlainOf target) )
  => (src -> tgt)  -- ^ the objective function
  -> DValue src
  -> ( target (TKScalar r)
     , DValue src )  -- morally DValue (ADTensorKind src)
{-# INLINE cgrad2_fwdS #-}
cgrad2_fwdS f x | Refl <- lemAppNil @sh =
  let g :: IxSOf target sh -> target (TKScalar r)
      g i = cjvp f x (soneHot (sscalar 1) i)
  in (kprimalPart $ f (fromDValue x), kbuild g)

fp :: (Floating a) => a -> a -> a -> a -> a
fp p1x p1y p2x p2y = (p1x ** 2 + p1y ** 2) - (p2x ** 2 + p2y ** 2)
  -- this is slower:
  -- fp p1x p1y p2x p2y = (sqr p1x + sqr p1y) - (sqr p2x + sqr p2y)

saddleGen
  :: (NumScalar a, Differentiable a, Ord (Concrete (TKScalar a)))
  => (Concrete (TKS '[2] a) -> (Concrete (TKScalar a), Concrete (TKS '[2] a)))
  -> (Concrete (TKS '[2] a) -> Concrete (TKS '[2] a)
      -> (Concrete (TKScalar a), Concrete (TKS '[2] a)))
  -> Concrete (TKS '[2] a)
  -> Concrete (TKS '[4] a)
{-# INLINE saddleGen #-}
saddleGen r1cost' r2cost' start =
  let r1 = multivariateArgmin magnitude_squaredS scaleS r1cost' start
      r2 = multivariateArgmax magnitude_squaredS scaleS (r2cost' r1) start
  in sappend r1 r2

-- TODO: this is very slow for many reasons:
-- * nested concrete derivatives are slow, because they need to nest ADVal
-- * we can't use symbolic derivatives due to the non-structured recursion
--   in multivariateMax
-- * we unroll all the identical things in the recursion and keep in memory
-- * nested derivatives in horde-ad are naively implemented regardless
-- * these are all 2-element rank 1 tensors (use products or lists instead?)
-- * fwd are slow, because they use Deltas instead of trivial dual numbers
-- * specialization and inlining is crucial here, but not investigated/forced
rr, ff, fr, rf :: Input -> Output
rr (Input (x, y)) = Nested.stoVector . unConcrete
                    $ saddleGen r1cost' r2cost' start
  where
    start = sfromListLinear (SNat @2 :$$ ZSS) [x, y]
    r1cost p1 = multivariateMax magnitude_squaredS scaleS
                                (r2cost' p1) (sfromPrimal start)
    r1cost' p1 = cgrad2 (r1cost) p1
    r2cost :: (NumScalar a, Differentiable a, ADReady target)
           => target (TKS '[2] a) -> target (TKS '[2] a)
           -> target (TKScalar a)
    r2cost r1 r2 = fp (r1 `sindex0` [0]) (r1 `sindex0` [1])
                      (r2 `sindex0` [0]) (r2 `sindex0` [1])
    r2cost' :: ( ADReadyNoLet target, ShareTensor target
               , ShareTensor (PrimalOf target), ShareTensor (PlainOf target)
               , NumScalar a, Differentiable a )
            => target (TKS '[2] a) -> target (TKS '[2] a)
            -> (target (TKScalar a), target (TKS '[2] a))
    r2cost' r1 r2 = cgrad2 (r2cost (sfromPrimal r1)) r2
ff (Input (x, y)) = Nested.stoVector . unConcrete
                    $ saddleGen r1cost' r2cost' start
  where
    start = sfromListLinear (SNat @2 :$$ ZSS) [x, y]
    r1cost p1 = multivariateMax magnitude_squaredS scaleS
                                (r2cost' p1) (sfromPrimal start)
    r1cost' p1 = cgrad2_fwdS (r1cost) p1
    r2cost :: (NumScalar a, Differentiable a, ADReady target)
           => target (TKS '[2] a) -> target (TKS '[2] a)
           -> target (TKScalar a)
    r2cost r1 r2 = fp (r1 `sindex0` [0]) (r1 `sindex0` [1])
                      (r2 `sindex0` [0]) (r2 `sindex0` [1])
    r2cost' :: ( ADReadyNoLet target, ShareTensor target
               , ShareTensor (PrimalOf target), ShareTensor (PlainOf target)
               , NumScalar a, Differentiable a )
            => target (TKS '[2] a) -> target (TKS '[2] a)
            -> (target (TKScalar a), target (TKS '[2] a))
    r2cost' r1 r2 = cgrad2_fwdS (r2cost (sfromPrimal r1)) r2
rf (Input (x, y)) = Nested.stoVector . unConcrete
                    $ saddleGen r1cost' r2cost' start
  where
    start = sfromListLinear (SNat @2 :$$ ZSS) [x, y]
    r1cost p1 = multivariateMax magnitude_squaredS scaleS
                                (r2cost' p1) (sfromPrimal start)
    r1cost' p1 = cgrad2 (r1cost) p1
    r2cost :: (NumScalar a, Differentiable a, ADReady target)
           => target (TKS '[2] a) -> target (TKS '[2] a)
           -> target (TKScalar a)
    r2cost r1 r2 = fp (r1 `sindex0` [0]) (r1 `sindex0` [1])
                      (r2 `sindex0` [0]) (r2 `sindex0` [1])
    r2cost' :: ( ADReadyNoLet target, ShareTensor target
               , ShareTensor (PrimalOf target), ShareTensor (PlainOf target)
               , NumScalar a, Differentiable a )
            => target (TKS '[2] a) -> target (TKS '[2] a)
            -> (target (TKScalar a), target (TKS '[2] a))
    r2cost' r1 r2 = cgrad2_fwdS (r2cost (sfromPrimal r1)) r2
fr (Input (x, y)) = Nested.stoVector . unConcrete
                    $ saddleGen r1cost' r2cost' start
  where
    start = sfromListLinear (SNat @2 :$$ ZSS) [x, y]
    r1cost p1 = multivariateMax magnitude_squaredS scaleS
                                (r2cost' p1) (sfromPrimal start)
    r1cost' p1 = cgrad2_fwdS (r1cost) p1
    r2cost :: (NumScalar a, Differentiable a, ADReady target)
           => target (TKS '[2] a) -> target (TKS '[2] a)
           -> target (TKScalar a)
    r2cost r1 r2 = fp (r1 `sindex0` [0]) (r1 `sindex0` [1])
                      (r2 `sindex0` [0]) (r2 `sindex0` [1])
    r2cost' :: ( ADReadyNoLet target, ShareTensor target
               , ShareTensor (PrimalOf target), ShareTensor (PlainOf target)
               , NumScalar a, Differentiable a )
            => target (TKS '[2] a) -> target (TKS '[2] a)
            -> (target (TKScalar a), target (TKS '[2] a))
    r2cost' r1 r2 = cgrad2 (r2cost (sfromPrimal r1)) r2
