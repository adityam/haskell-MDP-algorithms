-- (c) Aditya Mahajan <aditya.mahajan@mcgill.ca>
-- | Haskell library that implements the following algorithms to compute Gittins
-- index for a Markovian bandit:
--
-- 1. Largest remaining index algorithm proposed by 
--      
--      > Varaiya, Walrand, Buyyucuk, "Extensions of the multiarmed bandit
--      > problem: The discounted case", IEEE Transaction on Automatic Control,
--      > May 1985.
--
--      This is an off-line algorithm that gives the Gittins index of all
--      states.
--
--     
--              gittinsIndex :: Discount -> Bandit -> V.Vector Double
--      
--
-- 2. Restart algorithm proposed by
--
--      > Katehakis and Veinott, "The multiarmed bandit problem: Decomposition
--      > and computation", Mathematics of Operations Research, May 1987.
--
--      This is an on-line algorithm that gives the Gittins index (and the
--      stopping set) of one state.
--
--     
--              gittinsIndexAt :: Discount -> Bandit -> Int -> (Int, V.Vector Double)


module Bandit 
  ( -- Model
    Bandit(..), mkBandit
    -- Solution algorithms
  , gittinsIndex, gittinsIndexAt  
  ) where

import qualified Numeric.LinearAlgebra as N
import qualified Data.Vector as V
import qualified MDP as MDP
import Control.Arrow ( (&&&), (***) )
import Data.Maybe (fromJust)

data Bandit = Bandit
  { stateSize  :: !Int
  , reward     :: N.Vector Double
  , transition :: N.Matrix Double
  }

type Discount = Double

mkBandit :: Int -> [Double] -> [[Double]] -> Bandit
mkBandit stateSize_ reward_ transition_ = 
  Bandit stateSize_ (N.fromList reward_) (N.fromLists transition_)

-- Use Varaiya Walrand and Buyyucuk algorithm to calculate Gittins index.
gittinsIndex :: Discount -> Bandit -> V.Vector Double
gittinsIndex β (Bandit s r p) = go s nothing  where
    eye :: N.Matrix Double
    eye = N.ident s
    
    one :: N.Vector Double
    one = N.constant 1 s 

    nothing :: V.Vector (Maybe Double)
    nothing = V.replicate s Nothing

    n_inf :: Double
    n_inf = -1/0

    go :: Int             -- Step
       -> V.Vector (Maybe Double)  -- Collected gittins index
       -> V.Vector Double   -- Collected gittins index
    go n gittins = 
        let 
            {-
            -- Select columns for which the Gittins index is calucluated
            p_mask = N.fromList . V.toList . V.map (maybe 0 (const 1))     $ gittins

            -- Set the reward of states for which Gittins index is calculated to -Inf
            r_mask = N.fromList . V.toList . V.map (maybe 1 (const n_inf)) $ gittins
            -}
            -- Optimized version:
            (p_mask, r_mask) = (N.fromList . V.toList *** N.fromList . V.toList) 
                             . V.unzip . V.map (maybe (0,1) (const (1, n_inf)))
                             $ gittins

            prob = p N.<> (N.diag p_mask)
            discounted_p = eye - (N.scale β prob)

            expected_reward = (discounted_p N.<\> r ) * r_mask
            expected_time   = discounted_p N.<\> one

            (idx, state) = (N.maxElement &&& N.maxIndex) $ expected_reward / expected_time
            gittins' = gittins V.// [(state, Just idx)]
        in case n of
            0 -> V.map fromJust gittins
            _ -> go (n-1) gittins'

-- Use Katehakis and Veinott's "restart in x" algorithm to calculate Gittins index
gittinsIndexAt :: Discount -> Bandit -> Int -> (Double, V.Vector Int)
gittinsIndexAt β (Bandit s r p)  state = (idx, stopping_set) where
    mdp :: MDP.MDP
    mdp = MDP.MDP s
                  2 -- 0 action is continue, 1 action is restart
                  (V.fromList [r, N.constant (r N.@> state)  s])
                  (V.fromList [p, N.repmat (N.extractRows [state] p) s 1])
    (value, policy) = last $ MDP.policyIteration β mdp

    idx = (1-β) * (value N.@> state)
    stopping_set = V.findIndices ( == 1) policy
