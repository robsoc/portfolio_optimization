"""
The below script contains the logic of PortfolioOptimizer abstract superclass and corresponding subclassess than inherit
rebalancing and performance evaluation logic from the parent class.

Each child class adds the corresponding optimization logic to achieve the portfolio goal. The minimization problems
have been obtained through the mean of scipy optimizer.

author: Robert Soczewica
date: 15.06.2021
"""

import numpy as np
import pandas as pd

from collections import defaultdict
from abc import ABC, abstractmethod
from typing import List, Union, Dict, Tuple
from scipy import stats
from scipy.optimize import minimize



class PortfolioOptimizer(ABC):
    """
    Main portfolio optimization abstract superclass.

    Attributes
    ----------
    data : pd.DataFrame
        Closing asset levels
    wbound : Tuple[float]
        The weight bounds, defaults to (0, 1)
    pnav : pd.Series
        Portfolio Net Asset Value series
    pret : pd.Series
        Portfolio return series
    cum_pret : float
        Cumulative portfolio return 
    weights : pd.DataFrame
        Portfolio assets weight on rebalancing dates
    """    
    
    def __init__(self, data: pd.DataFrame, wbound: Tuple[float]=(0, 1)) -> None:
        self.data = data
        self.wbound = wbound
        self._pret = None
        self._cum_pret = None
        
        
    @abstractmethod    
    def optimize(self, ret: pd.DataFrame) -> np.array:
        """
        Abstract optimization method that needs to replaced in child classess

        Parameters
        ----------
        ret : pd.DataFrame
            Return data

        Returns
        -------
        np.array
            Assets weights
        """        
        pass
    
    
    def run(self, window: int=None) -> None:
        """
        Runs the portfolio optimization and rebalancing.

        Parameters
        ----------
        window : int, optional
            Rebalancing window, by default None
        """        
        
        # Calculate simple assets' returns
        ret = self.data.pct_change().dropna()
        
        if window is not None:
            # Initial conditions
            units = 0
            pnav = pd.Series(dtype='float64')
            weights = {}
            
            # Rebalance portfolio with a given window
            for i in range(window, len(ret), window):
                # Caluclate past (last window) portfolio NAV
                tmp = ret.iloc[i-window:i, :]
                past_pnav = (units * self.data.iloc[i-window:i, :]).sum(axis=1)
                pnav = pnav.append(past_pnav)
                
                # Optimize portfolio weights and determine units of assets to buy
                w = self.optimize(ret=tmp)
                units = w * self.data.iloc[i, :].sum() / self.data.iloc[i, :]
                
                # Save portfolio weights at rebalance periods in an attribute
                weights[units.name] = w
            
            # Calculate current (last rebalance) portfolio NAV
            curr_pnav = (units * self.data.iloc[i:, :]).sum(axis=1)
            self.pnav = pnav.append(curr_pnav).replace(0, np.nan)
            
            # Store portfolio weights at rebalancing dates
            self.weights = pd.DataFrame.from_dict(weights, orient='index', columns=self.data.columns)
            
        else:
            # Calculate weights and portfolio NAV without rebelancing
            self.weights = self.optimize(ret=ret)
            self.pnav = (self.weights * self.data).sum(axis=1)


    @staticmethod
    def cum_ret(ret: pd.Series, total: bool=True) -> Union[float, pd.Series]:
        """
        Calculates cumulative return of a series.

        Parameters
        ----------
        ret : pd.Series
            Return data
        total : bool, optional
            Determines if whether to show total return or series, by default True

        Returns
        -------
        Union[float, pd.Series]
            Total cumulative return or cumulative return series
        """        
        out = (1 + ret).cumprod() - 1
        return out[-1] if total else out
    
    
    @property
    def pret(self) -> pd.Series:
        if self._pret is None:
            self._pret = self.pnav.pct_change().dropna()
        return self._pret
    
    
    @property
    def cum_pret(self) -> float:
        if self._cum_pret is None:
            self._cum_pret = self.cum_ret(self.pret)
        return self._cum_pret
    
    
    @classmethod
    def cagr(cls, nav: pd.Series) -> float:
        """
        Compound Annual Growth Rate.

        Parameters
        ----------
        nav : pd.Series
            Net Asset Value of an asset

        Returns
        -------
        float
            CAGR metric
        """        
        nav.dropna(inplace=True)
        nyears = len(nav.index.year.unique())
        return (nav[-1]/nav[0])**(1/nyears) - 1
    
    
    @staticmethod
    def ann_vol(ret: pd.Series) -> float:
        """
        Annualised volatility.

        Parameters
        ----------
        ret : pd.Series
            Return data

        Returns
        -------
        float
            Annualised volatility metric
        """        
        return np.std(ret) * np.sqrt(12)

    
    @staticmethod
    def sharpe_ratio(ret: pd.Series) -> float:
        """
        Sharpe ratio.

        Parameters
        ----------
        ret : pd.Series
            Return data

        Returns
        -------
        float
            Sharpe ratio metric
        """        
        return np.mean(ret) / np.std(ret)


    @staticmethod
    def sortino_ratio(ret: pd.Series, dthres: float=0) -> float:
        """
        Sortino ratio.

        Parameters
        ----------
        ret : pd.Series
            Return data
        dthres : float, optional
            Downside threshold, by default 0

        Returns
        -------
        float
            Sortino ratio metric
        """        
        return np.mean(ret) / np.std(ret[ret < dthres])
    
    
    def capm(self, bret: pd.Series) -> Tuple[float]:
        """
        Capital Asset Pricing Model (CAPM)

        Parameters
        ----------
        bret : pd.Series
            Benchmark return data

        Returns
        -------
        Tuple[float]
            Beta and Alpha metrics
        """        
        return stats.linregress(bret.values, self.pret.values)[0:2]
    
    def info_ratio(self, bret: pd.Series) -> float:
        """
        Information ratio.

        Parameters
        ----------
        bret : pd.Series
            Benchmark return data

        Returns
        -------
        float
            Information ratio metric
        """        
        return np.mean(self.pret - bret) / np.std(self.pret - bret)


    def treynor_ratio(self, bret: pd.Series, beta: float=None) -> float:
        """
        Treynor ratio.

        Parameters
        ----------
        bret : pd.Series
            Benchmark return data
        beta : float, optional
            Beta of portfolio, by default None

        Returns
        -------
        float
            Treynor ratio metric
        """        
        beta = self.capm(bret)[0] if beta is None else beta
        return np.mean(self.pret) / beta
    
    
    def summary(self, bdata: pd.Series=None) -> pd.DataFrame:
        """
        Portfolio performance summary.

        Parameters
        ----------
        bdata : pd.Series, optional
            Benchmark close level data, by default None

        Returns
        -------
        pd.DataFrame
            Table with performance metrics
        """        
        
        d = {}
        metrics = ["Cumulative Return", "CAGR", "Annualised Vol", "Sharpe Ratio", "Sortino Ratio"]
        d["Portfolio"] = [self.cum_pret, self.cagr(self.pnav), self.ann_vol(self.pret), self.sharpe_ratio(self.pret), self.sortino_ratio(self.pret)]
                
        if bdata is not None:
            bret = bdata.pct_change().dropna()
            d["Benchmark"] = [self.cum_ret(bret), self.cagr(bdata), self.ann_vol(bret), self.sharpe_ratio(bret), self.sortino_ratio(bret)]
            metrics += ["Jensen's Alpha", "Beta", "Information Ratio", "Treynor Ratio"]
            beta, alpha = self.capm(bret)
            d["Portfolio"] += [alpha, beta, self.info_ratio(bret), self.treynor_ratio(bret, beta)]
            
        return pd.DataFrame.from_dict(d, orient='index', columns=metrics).T
  
    

class EquityBond(PortfolioOptimizer):
    """
    Child class. Creates Equity-Bond portfolio.

    Attributes
    ----------
    equity_col : str
        Name of the column with equity asset
    bond_col : str
        Name of the column with bond asset
    equity_pct : float
        Percentage of portfolio allocated to equity class, defaults to 60
    """    
    
    def __init__(self, data: pd.DataFrame, equity_col: str, bond_col: str, equity_pct: float=60) -> None:
        self.equity_col = equity_col
        self.bond_col = bond_col
        self.equity_pct = equity_pct
        super().__init__(data=data[[self.equity_col, self.bond_col]])
        
        
    def optimize(self, ret: pd.DataFrame) -> np.array:  
        """
        Portfolio optimization logic.

        Parameters
        ----------
        ret : pd.DataFrame
            Return data

        Returns
        -------
        np.array
            Weights
        """        
              
        return np.array([self.equity_pct/100, 1 - self.equity_pct/100])
        
       
        
class EqualWeight(PortfolioOptimizer):
    """
    Child class. Creates Equally Weighted Portfolio (EWP).
    """    
    
    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__(data=data)
        
        
    def optimize(self, ret: pd.DataFrame) -> np.array:
        """
        EWP optimization logic.

        Parameters
        ----------
        ret : pd.DataFrame
            Return data

        Returns
        -------
        np.array
            Weights
        """  
             
        return np.repeat(1/ret.shape[1], ret.shape[1])
    
    
        
class MinVar(PortfolioOptimizer):
    """
    Child class. Creates Minimum Variance Portfolio (MVP).
    """    
    
    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__(data)
    
    
    @staticmethod
    def pvar(w: np.array, V: np.array) -> float:
        """
        Calculates portfolio variance.

        Parameters
        ----------
        w : np.array
            Weights
        V : np.array
            Covariance matrix

        Returns
        -------
        float
            Portfolio variance
        """        
        
        return w.dot(V).dot(w)
    
    
    def optimize(self, ret: pd.DataFrame) -> np.array:
        """
        MVP optimization logic. Minimizes portfolio variance.

        Parameters
        ----------
        ret : pd.DataFrame
            Return data

        Returns
        -------
        np.array
            Weights
        """        
        
        # Initial guess of weights
        w_0 = np.repeat(1/ret.shape[1], ret.shape[1])
        
        # Portfolio assets covariance
        V = np.cov(ret, rowvar=False)
        
        # Optimization constraints
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = [self.wbound] * ret.shape[1] if self.wbound is not None else None
        
        # Portfolio variance minimization
        w = minimize(self.pvar, w_0, args=V, bounds=bounds, method='SLSQP', constraints=cons).x
        
        return w
    


class ERC(MinVar):
    """
    Child class. Creates Equal Risk Contribution (ERC) portfolio.
    """    
    
    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__(data)
    
        
    def risk_contrib(self, w: np.array, V: np.array) -> np.array:
        """
        Calculates risk contribution of assets.

        Parameters
        ----------
        w : np.array
            Weights
        V : np.array
            Covariance matrix

        Returns
        -------
        np.array
            Risk contributions
        """        
        
        sig = np.sqrt(self.pvar(w, V))
        mrc = w.dot(V)
        return w * mrc / sig
    
    def obj_func(self, w: np.array, V: np.array) -> float:
        """
        Optimization objective function.

        Parameters
        ----------
        w : np.array
            Weights
        V : np.array
            Covariance matrix

        Returns
        -------
        float
            Objective function value
        """        
        sig = np.sqrt(self.pvar(w, V)) 
        risk_tgt = sig * w
        rc = self.risk_contrib(w, V)
        return np.sum((rc - risk_tgt)**2)
        
        
    def optimize(self, ret: pd.DataFrame) -> np.array:
        """
        ERC optimization logic. 

        Parameters
        ----------
        ret : pd.DataFrame
            Return data

        Returns
        -------
        np.array
            Weights
        """        
        # Initial guess of weights
        w_0 = np.repeat(1/ret.shape[1], ret.shape[1])
        
        # Portfolio assets covariance
        V = np.cov(ret, rowvar=False)
        
        # Optimization constraints
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = [self.wbound] * ret.shape[1] if self.wbound is not None else None
        
        # Objective function minimization
        w = minimize(self.obj_func, w_0, args=V, bounds=bounds, method='SLSQP', constraints=cons, options={'ftol': 1e-12}).x

        return w



class TargetReturn(MinVar):
    """
    Child class. Performs mean-variance optimization for target return.

    Attributes
    ----------
    target : float
        Target portfolio return
    """    
    
    def __init__(self, data: pd.DataFrame, target: float=0.05) -> None:
        self.target = target
        super().__init__(data)
    
    
    def ann_pvol(self, w: np.array, V: np.array) -> float:
        """
        Annulised portfolio volatility.

        Parameters
        ----------
        w : np.array
            Weights
        V : np.array
            Covariance matrix

        Returns
        -------
        float
            Annualised volatility
        """        
        
        return np.sqrt(self.pvar(w, 12*V)) 
        
        
    def optimize(self, ret: pd.DataFrame, out: str='x') -> Union[float, np.array]:
        """
        Mean-Variance optimization logic.

        Parameters
        ----------
        ret : pd.DataFrame
            Return data
        out : str, optional
            Optimization output shape, by default 'x' (weights)

        Returns
        -------
        Union[float, np.array]
            Optimization output, can be weights, obj function value etc.
        """        
        
        # Initial guess of weights
        w_0 = np.repeat(1/ret.shape[1], ret.shape[1])
        
        # Portfolio assets covariance and mean return
        V = np.cov(ret, rowvar=False)
        mu_ret = np.mean(ret)
        
        # Optimization constraints
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.sum(x*mu_ret)*12 - self.target})
        bounds = [self.wbound] * ret.shape[1] if self.wbound is not None else None
        
        # Annual volatility minimization for given target return
        w = minimize(self.ann_pvol, w_0, args=V, bounds=bounds, method='SLSQP', constraints=cons)
                
        return getattr(w, out)