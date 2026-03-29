{-# LANGUAGE OverloadedLists, OverloadedStrings #-}
module GradBench.Saddle (Input, Output, rr, ff, fr, rf) where

import Data.Aeson ((.:))
import Data.Aeson qualified as JSON
import Data.Array.Nested qualified as Nested
import Data.Vector.Storable qualified as VS
import GHC.TypeLits (KnownNat)
import GradBench.GD
import HordeAd
import HordeAd.Core.Adaptor

data Input = Input (Double, Double)

type Output = VS.Vector Double

instance JSON.FromJSON Input where
  parseJSON = JSON.withObject "input" $ \o ->
    Input <$> o .: "start"

cgrad2_fwdR
  :: forall src r tgt target n.
     ( src ~ ADVal target (TKR n r)
     , NumScalar r, ADTensorScalar r ~ r, KnownNat n
     , tgt ~ ADVal target (TKScalar r)
     , ADReadyNoLet target, ShareTensor target
     , ShareTensor (PrimalOf target), ShareTensor (PlainOf target) )
  => (src -> tgt)  -- ^ the objective function
  -> DValue src
  -> ( target (TKScalar r)
     , DValue src )  -- morally DValue (ADTensorKind src)
{-# INLINE cgrad2_fwdR #-}
cgrad2_fwdR f x =
  let shr = rshape $ fromDValue @src x
      g :: IxROf target n -> target (TKScalar r)
      g i = cjvp f x (roneHot shr (rscalar 1) i)
  in (kprimalPart $ f (fromDValue x), rbuild shr (rfromK . g))
{- TODO: optimize by switching the signature to shaped and using s simplified
   version of this (or switch to symbolic pipeline):
  in withShsFromShR shr $ \(sh :: ShS sh)->
     withKnownShS sh $
     case lemAppNil @sh  of
       Refl ->
         let g :: IxSOf target sh -> target (TKScalar r)
             g i = cjvp f x (rfromS $ soneHot @sh (sscalar 1) i)
         in (kprimalPart $ f (fromDValue x), rfromS $ kbuild @sh g) -}

fp :: (Floating a) => a -> a -> a -> a -> a
fp p1x p1y p2x p2y = (p1x ** 2 + p1y ** 2) - (p2x ** 2 + p2y ** 2)
  -- this is slower:
  -- fp p1x p1y p2x p2y = (sqr p1x + sqr p1y) - (sqr p2x + sqr p2y)

saddleGen
  :: (NumScalar a, Differentiable a, Ord (Concrete (TKScalar a)))
  => (Concrete (TKR 1 a) -> (Concrete (TKScalar a), Concrete (TKR 1 a)))
  -> (Concrete (TKR 1 a) -> Concrete (TKR 1 a)
      -> (Concrete (TKScalar a), Concrete (TKR 1 a)))
  -> Concrete (TKR 1 a)
  -> Concrete (TKR 1 a)
{-# INLINE saddleGen #-}
saddleGen r1cost' r2cost' start =
  let r1 = multivariateArgmin magnitude_squaredR scaleR r1cost' start
      r2 = multivariateArgmax magnitude_squaredR scaleR (r2cost' r1) start
  in rappend r1 r2

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
rr (Input (x, y)) = Nested.rtoVector . unConcrete
                    $ saddleGen r1cost' r2cost' start
  where
    start = rfromList [rscalar x, rscalar y]
    r1cost p1 = multivariateMax magnitude_squaredR scaleR
                                (r2cost' p1) (rfromPrimal start)
    r1cost' p1 = cgrad2 (r1cost) p1
    r2cost :: forall target a. (NumScalar a, Differentiable a, ADReady target)
           => target (TKR 1 a) -> target (TKR 1 a)
           -> target (TKScalar a)
    r2cost r1 r2 = fp (r1 `rindex0` [0]) (r1 `rindex0` [1])
                      (r2 `rindex0` [0]) (r2 `rindex0` [1])
    r2cost' :: forall target a.
               ( ADReadyNoLet target, ShareTensor target
               , ShareTensor (PrimalOf target), ShareTensor (PlainOf target)
               , NumScalar a, Differentiable a )
            => target (TKR 1 a) -> target (TKR 1 a)
            -> (target (TKScalar a), target (TKR 1 a))
    r2cost' r1 r2 = cgrad2 @_ @_ @_ @target (r2cost (rfromPrimal r1)) r2
ff (Input (x, y)) = Nested.rtoVector . unConcrete
                    $ saddleGen r1cost' r2cost' start
  where
    start = rfromList [rscalar x, rscalar y]
    r1cost p1 = multivariateMax magnitude_squaredR scaleR
                                (r2cost' p1) (rfromPrimal start)
    r1cost' p1 = cgrad2_fwdR (r1cost) p1
    r2cost :: forall target a. (NumScalar a, Differentiable a, ADReady target)
           => target (TKR 1 a) -> target (TKR 1 a)
           -> target (TKScalar a)
    r2cost r1 r2 = fp (r1 `rindex0` [0]) (r1 `rindex0` [1])
                      (r2 `rindex0` [0]) (r2 `rindex0` [1])
    r2cost' :: forall target a.
               ( ADReadyNoLet target, ShareTensor target
               , ShareTensor (PrimalOf target), ShareTensor (PlainOf target)
               , NumScalar a, Differentiable a )
            => target (TKR 1 a) -> target (TKR 1 a)
            -> (target (TKScalar a), target (TKR 1 a))
    r2cost' r1 r2 = cgrad2_fwdR @_ @_ @_ @target (r2cost (rfromPrimal r1)) r2
rf (Input (x, y)) = Nested.rtoVector . unConcrete
                    $ saddleGen r1cost' r2cost' start
  where
    start = rfromList [rscalar x, rscalar y]
    r1cost p1 = multivariateMax magnitude_squaredR scaleR
                                (r2cost' p1) (rfromPrimal start)
    r1cost' p1 = cgrad2 (r1cost) p1
    r2cost :: forall target a. (NumScalar a, Differentiable a, ADReady target)
           => target (TKR 1 a) -> target (TKR 1 a)
           -> target (TKScalar a)
    r2cost r1 r2 = fp (r1 `rindex0` [0]) (r1 `rindex0` [1])
                      (r2 `rindex0` [0]) (r2 `rindex0` [1])
    r2cost' :: forall target a.
               ( ADReadyNoLet target, ShareTensor target
               , ShareTensor (PrimalOf target), ShareTensor (PlainOf target)
               , NumScalar a, Differentiable a )
            => target (TKR 1 a) -> target (TKR 1 a)
            -> (target (TKScalar a), target (TKR 1 a))
    r2cost' r1 r2 = cgrad2_fwdR @_ @_ @_ @target (r2cost (rfromPrimal r1)) r2
fr (Input (x, y)) = Nested.rtoVector . unConcrete
                    $ saddleGen r1cost' r2cost' start
  where
    start = rfromList [rscalar x, rscalar y]
    r1cost p1 = multivariateMax magnitude_squaredR scaleR
                                (r2cost' p1) (rfromPrimal start)
    r1cost' p1 = cgrad2_fwdR (r1cost) p1
    r2cost :: forall target a. (NumScalar a, Differentiable a, ADReady target)
           => target (TKR 1 a) -> target (TKR 1 a)
           -> target (TKScalar a)
    r2cost r1 r2 = fp (r1 `rindex0` [0]) (r1 `rindex0` [1])
                      (r2 `rindex0` [0]) (r2 `rindex0` [1])
    r2cost' :: forall target a.
               ( ADReadyNoLet target, ShareTensor target
               , ShareTensor (PrimalOf target), ShareTensor (PlainOf target)
               , NumScalar a, Differentiable a )
            => target (TKR 1 a) -> target (TKR 1 a)
            -> (target (TKScalar a), target (TKR 1 a))
    r2cost' r1 r2 = cgrad2 @_ @_ @_ @target (r2cost (rfromPrimal r1)) r2
