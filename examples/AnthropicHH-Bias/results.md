using bias loss as a performance function:
1. loss on base 410m model: 0.513
2. loss on full sft: 0.4871
3. loss on random 45k samples: 0.4987
4. loss on keeping lowest 45k ekfac scores: 0.5965
5. loss on keeping highest 45k ekfac scores: 0.3727 


You can find the SFTed model LORA adapters at: 
ncgc/pythia_410m_hh_full_sft_trainer
ncgc/pythia_410m_sft_hh_random_45k
ncgc/pythia_410m_sft_hh_45k_lowest.bias
ncgc/pythia_410m_sft_hh_45k_highest.bias
