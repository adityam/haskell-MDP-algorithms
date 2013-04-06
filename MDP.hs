module MDP 
  ( -- Model
    MDP (..) , mkMDP
    -- Types
  , QFunction , VFunction , Policy , ValuePolicyPair
  , qUpdate , vUpdate , bellmanUpdate 
  , valueIteration , valueIterationN
  ) where

import qualified Data.Vector as V
import qualified Numeric.LinearAlgebra as N
import Control.Arrow ( (&&&), (***) )

-- Data type to model a finite Markov Decision Process (MDP). 
--
-- Let $S = {\tt stateSize}$ and $A = {\tt actionSize}$ denote the size of the
-- state and action spaces respectively. 
--
-- `reward` is a $S×A$ matrix, and `transition` is a collection of $A$
-- matrices, each of size $S×S$.

data MDP = MDP 
   { stateSize  :: !Int
   , actionSize :: !Int
   , reward     :: V.Vector (N.Vector Double)
   , transition :: V.Vector (N.Matrix Double)
   }

mkMDP :: Int -> Int -> [[Double]] -> [[[Double]]] -> MDP
mkMDP stateSize_ actionSize_ reward_ transition_ = 
    MDP stateSize_ actionSize_ 
        (V.fromList . map N.fromList  $ reward_)
        (V.fromList . map N.fromLists $ transition_)

-- TODO:
-- check :: Function to check if the model is correct

type QFunction = N.Matrix Double
type VFunction = N.Vector Double
type Policy    = V.Vector Int

type ValuePolicyPair = (VFunction, Policy)

type Discount = Double

-- In vectorized form, the Q-update is given by
-- 
-- $$Q(⋅,u) = r(⋅,u) + β P(u) V(⋅)$$
-- 
-- We micro-optimize the implementation by rearranging terms as
--
-- $$Q(⋅,u) = r(⋅,u) + P(u) (β V(⋅))$$
--
-- In the following code, the second term of the summand is called
-- `expectedCostToGo`. 
--
-- All collection of vectors and matrices are stored as `Data.Vector`. The hope
-- is that this will make it easy to parallelize the algorithm in the future.
--
-- The final result is converted to a `Numeric.LinearAlgebra.Matrix` because we
-- later use the `maxElement` and `maxIndex` functions from
-- `Numeric.LinearAlgebra`.

{-# INLINE qUpdate #-}
qUpdate :: Discount -> MDP -> ValuePolicyPair -> QFunction
qUpdate β mdp (vFunction,_) = N.fromColumns . V.toList $ qMatrix
    where
    rFunction = reward mdp     -- The collection of vectors  $\{ r(⋅, u) | u \in A \}$
    pMatrices = transition mdp -- The collection of matrices $\{ P(u) | u \in A \}$

    scaledVFunction = N.scale β vFunction -- See note about micro-optimization above

    -- TODO: Parallelize the computation of expectedCostToGo
    expectedCostToGo = V.map (N.<> scaledVFunction) pMatrices -- This is the second term of the summand
    qMatrix  = V.zipWith (+) rFunction expectedCostToGo

-- Value update is given by
-- 
-- $$ V(x) = max \{ Q(x,u) : u \in U \} $$

{-# INLINE vUpdate #-}
vUpdate :: QFunction -> ValuePolicyPair
vUpdate  = (N.fromList *** V.fromList) . unzip . map (N.maxElement &&& N.maxIndex) . N.toRows

-- Bellman update is the composition of `vUpdate` and `qUpdate`.

{-# INLINE bellmanUpdate #-}
bellmanUpdate :: Discount -> MDP -> ValuePolicyPair -> ValuePolicyPair
bellmanUpdate β mdp = vUpdate . qUpdate β mdp 

-- Start with the all zeros vector and run `bellmanUpdate` until the `spanNorm`
-- of successive terms with within $ε(1-β)/β$ of each other. This ensures that
-- the renormalized value function is within $ε$ of the optimal solution.

valueIteration :: Double -> Discount -> MDP -> [ValuePolicyPair]
valueIteration ε β mdp = 
    let 
        k = if β < 1 
               then (1-β)/β
               else 1
        δ = k*ε

        initial :: ValuePolicyPair 
        initial = (N.konst 0 (stateSize mdp), V.replicate (actionSize mdp) 0)

        bellmanUpdate' :: ValuePolicyPair -> ValuePolicyPair
        bellmanUpdate' = bellmanUpdate β mdp

        iterations :: [ValuePolicyPair]
        iterations = iterate bellmanUpdate' initial

        pairedIteration :: [(ValuePolicyPair, ValuePolicyPair)]
        pairedIteration = zip iterations (tail iterations)

        stoppingRule :: (ValuePolicyPair, ValuePolicyPair) -> Bool
        stoppingRule ((v1,_), (v2,_)) = spanNorm v1 v2 < δ

        -- See Puterman 6.6.12 for details
        renormalize :: (ValuePolicyPair, ValuePolicyPair) -> ValuePolicyPair
        renormalize ( (v1, _), (v2, p2) ) = let w = N.maxElement (v2-v1)
                                            in (v2 + N.scalar (w/k), p2)

     in map renormalize . takeWhile (not . stoppingRule) $ pairedIteration

valueIterationN:: Discount -> Double -> MDP -> [(Int, ValuePolicyPair)]
valueIterationN β ε mdp = zip [2..] (valueIteration β ε mdp)

{-# INLINE spanNorm #-}
spanNorm :: N.Vector Double -> N.Vector Double -> Double
spanNorm u v = N.maxElement w - N.minElement w
      where w = u - v
