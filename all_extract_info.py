
#EXTRACT CH NUM, CH POS, VARIANT TYPE, GENE NAME FOR ALL FILES 

import os
import csv

def extract_information(annotations):
    # Split the ANN field to get the relevant information
    fields = annotations.split('|')
    
    # Check if the fields have the expected structure
    if len(fields) >= 6:
        variant_type = fields[1]
        gene_name = fields[4]
        return variant_type, gene_name
    else:
        # Return None if the expected fields are not found
        return None, None

def process_gvcf(input_file, output_file, allowed_genes):
    found_genes = set()
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        # Create a CSV writer
        writer = csv.writer(outfile, delimiter='\t')
        
        # Write header to the output file
        writer.writerow(['Chromosome', 'Position', 'VariantType', 'GeneName'])
        
        # Process each line in the input file
        for line in infile:
            # Skip header lines
            if line.startswith('#'):
                continue
            
            # Split the line into fields
            fields = line.strip().split('\t')
            
            # Check if the line has enough fields
            if len(fields) > 7:
                # Extract chromosome and position
                chromosome = fields[0]
                position = fields[1]
                
                # Extract annotations field
                annotations = fields[7].split(';')[0]
                
                # Extract information
                variant_type, gene_name = extract_information(annotations)
                
                # Check if the gene is in the allowed_genes dictionary
                if gene_name in associated_genes:
                    found_genes.add(gene_name)
                    # Write to the output file if information is available
                    if all([chromosome, position, variant_type, gene_name]):
                        writer.writerow([chromosome, position, variant_type, gene_name])
    
    # Collect genes from the dictionary that were not found in the VCF file
    not_added_genes = allowed_genes - found_genes
    


# Example usage
# Replace placeholders with your actual file paths



# Dictionary of gene names
associated_genes = {'ABAT','ABCA10','ABCA13','ABCA7','ABCE1','ABL2','ACE','ACHE','ACTB','ACTL6B',
                'ACTN4','ACY1','ADA','ADCY3','ADCY5','ADGRL1','ADK','ADNP','ADORA3','ADSL',
                'ADSS2','AFF2','AGAP1','AGAP2','AGAP5','AGBL4','AGMO','AGO1','AGO2','AGO3',
                'AGO4','AGTR2','AHDC1','AHI1','AHNAK','AKAP9','ALDH1A3','ALDH1L1','ALDH5A1',
                'ALG6','AMPD1','AMT','ANK2','ANK3','ANKRD11','ANKRD17','ANKS1B','INP32A',
                'ANXA1','AP1S2','AP2M1','AP2S1','APBA2','APBB1','APH1A','ARF3','ARHGAP11B',
                'ARHGAP32','ARHGAP5','ARHGEF10','ARHGEF2','ARHGEF9','ARID1B','ARID2','ARNT2',
                'ARX','ADORA2A','ADRB2','ASAP2','ASB14','ASH1L','AR','ASMT','ASPM','ASTN2',
                'ASXL3','ATP10A','ATP1A1','ATP1A3','ATP2B1','ATP2B2','ATP6V0A2','ATRX',
                'AUTS2','AVPR1A','AZGP1','BACE1','BAIAP2L1','BAZ2B','BBS4','BCAS1','BCKDK',
                'BCL11A','BCORL1','BICRA','BIRC6','BRAF','BRCA2','BRD4',
                'BRINP3', 'BRSK2', 'BRWD3', 'BST1', 'BTAF1', 'BTRC', 'C12orf57', 'C15orf62',
                'AVPR1B', 'BCL11B', 'BICDL1', 'C4B', 'CA6', 'CACNA1A', 'CACNA1B', 'CACNA1C',
                'CACNA1D', 'CACNA1E', 'CACNA1F', 'CACNA1G', 'CACNA1H', 'CACNA1I', 'CACNA2D1',
                'CACNA2D3', 'CACNB1', 'CACNB2', 'CACNG2', 'CADM1', 'CADM2', 'CADPS', 'CADPS2',
                'CAMK2A', 'CAMK2B', 'CAMK4', 'CAMTA2', 'CAPN12', 'CAPRIN1', 'CARD11', 'CASK',
                'CASKIN1', 'CASZ1', 'CAT', 'CBX1', 'CC2D1A', 'CCDC88C', 'CCDC91', 'CCIN',
                'CCNG1', 'CCNK', 'CCSER1', 'CCT4', 'CD276', 'CD38', 'CD99L2', 'CDC42BPB',
                'CDH10', 'CDH11', 'CDH13', 'CDH2', 'CDH8', 'CDH9', 'CDK13', 'CDK16', 'CDK19',
                'CDK5RAP2', 'CDK8', 'CDKL5', 'CDON', 'CECR2', 'CELF2', 'CELF4', 'CELF6', 'CEP135',
                'CEP290', 'CEP41', 'CERT1', 'CGNL1','CHAMP1', 'CHD1', 'CHD2', 'CHD3', 'CHD7',
                'CHD8', 'CHD9', 'CHKB', 'CHM', 'CHMP1A', 'CHRM3', 'CHRNA7', 'CHRNB3', 'CHST2',
                'CIB2', 'CIC', 'CLASP1', 'CLCN4', 'CLN8', 'CLTCL1', 'CMIP', 'CMPK2', 'CNGB3',
                'CNKSR2','CNOT1', 'CNOT3', 'CNR1', 'CNTN3', 'CNTN4', 'CNTN5', 'CNTN6', 'CNTNAP2', 
                'CDH22', 'CLIP2', 'CNTNAP3', 'CNTNAP4', 'CNTNAP5', 'COL28A1', 'CORO1A', 'CPEB4', 
                'CPSF7', 'CPT2', 'CPZ', 'CREBBP', 'CSDE1', 'CSMD1', 'CSMD3', 'CSNK1E', 'CSNK1G1', 
                'CSNK2A1', 'CSNK2B', 'CTCF', 'CTNNA2', 'CTNNA3', 'CTNNB1', 'CTNND2', 'CTR9', 'CTTNBP2', 
                'CUL3', 'CUL7', 'CUX1', 'CUX2', 'CX3CR1', 'CYFIP1', 'CYLC2', 'CYP27A1', 'DAGLA', 'DAPP1', 
                'DCC', 'DDC', 'DDHD2', 'DDX23', 'DDX3X', 'DDX53', 'DEAF1', 'DENR', 'DEPDC5', 'DGKI', 'DHCR7',
                'DHX30', 'DHX9', 'DIP2A', 'DIP2C', 'DIPK2A','DISC1', 'DIXDC1',' DLG1', 'DLG2', 'DLG4', 
                'DLGAP1', 'DLGAP2', 'DLGAP3', 'DLL1', 'DLX3' ,'DLX6', 'DMD',' DMPK',' DMWD',' DMXL2','DNAH10', 
                'DNAH17',' DNAH3' ,'DNER',' DNMT3A',' CYP11B1',' DLX2',' DNM1',' DOCK1 ','DOCK4' ,'DOCK8',
                'DOLK ','DPP10', 'DPP3','DPP4','DPP6','DPYD', 'DPYSL2', 'DPYSL3','DRD1', 'DRD2', 'DRD3', 
                'DSCAM', 'DST', 'DUSP15', 'DVL3', 'DYDC1', 'DYDC2', 'DYNC1H1', 'DYRK1A', 'EBF3', 'ECPAS', 
                'EEF1A2', 'EFR3A', 'EGR3', 'EHMT1', 'EIF3G', 'EIF4E', 'EIF4G1', 'ELAVL2', 'ELAVL3', 
                'ELOVL2', 'ELP2', 'ELP4', 'EMSY', 'EN2', 'ENPP1', 'EP300', 'EP400',
                'EPC2', 'EPHA1', 'EPHB1', 'EPHB2', 'EPPK1', 'ERBIN', 'ERMN', 'ESR2', 'ESRRB', 'ETFB',
                'EXOC3', 'EXOC5', 'EXOC6', 'EXOC6B', 'EXT1', 'FABP4', 'FABP5', 'FAM47A', 'FAM98C', 'FAN1',
                'FAT1', 'EIF5A', 'ERG', 'FBN1', 'FBRSL1', 'FBXL13', 'FBXO11', 'FBXO33', 'FBXO40', 'FCRL6',
                'FEZF2', 'FGA', 'FGF13', 'FGF14', 'FGFR1', 'FHIT', 'FLNA', 'FMR1', 'FOXG1', 'FOXP1',
                'FOXP2', 'FRG1', 'FRK', 'FRMPD4', 'FXN', 'G3BP2', 'GABBR2', 'GABRA3', 'GABRA4', 'GABRB2',
                'GABRB3', 'GABRG2', 'GABRG3', 'GALNT10', 'GALNT13', 'GALNT14', 'GALNT2', 'GALNT8', 'GAS2',
                'GATM', 'GBE1', 'GDA', 'GFAP', 'GGNBP2', 'GIGYF1', 'GIGYF2', 'GLIS1', 'GLO1', 'GLRA2',
                'GNAI1', 'GNAS', 'GNB1L', 'GNB2', 'GPC4', 'GPC5','GPC6', 'GPD2', 'GPHN', 'GPR37', 'GPR85', 
                'GPX1', 'GRB10', 'GRIA1', 'GRIA2', 'GRIA3',
                'GRID1', 'GRID2', 'GRID2IP', 'GRIK2', 'GRIK3', 'GRIK4', 'GRIK5', 'GRIN1', 'GRIN2A', 'GRIN2B',
                'GRIP1', 'GRK4', 'GRM5', 'GRM7', 'GSTM1', 'GTF2I', 'GUCY1A2', 'H1-4', 'H2BC11', 'H4C11',
                'H4C3', 'H4C5', 'HACE1', 'HCFC1', 'HCN1', 'HDAC4', 'HDAC8', 'HDLBP', 'HECTD4', 'HECW2',
                'HEPACAM', 'HERC1', 'HERC2', 'HIVEP2', 'HIVEP3', 'HLA-A', 'HLA-B', 'HLA-DPB1', 'HLA-DRB1',
                'HLA-G', 'HMGN1', 'HNRNPD', 'HNRNPF', 'HNRNPH2', 'HNRNPK', 'HNRNPR', 'HNRNPU', 'HNRNPUL2',
                'HOMER1', 'HOXA1', 'HRAS', 'HS3ST5', 'HSD11B1', 'HTR1B', 'HTR3A', 'HTR3C', 'HUWE1', 'HYDIN',
		'ICA1', 'IGF1', 'IKZF1', 'IL1R2', 'IL1RAPL1', 'IL1RAPL2', 'ILF2', 'IMMP2L', 'INPP1', 'INTS1',
		'INTS6', 'IQGAP3', 'IQSEC2', 'IRF2BPL', 'ITGA8', 'ITGB3', 'ITPR1', 'ITSN1', 'JARID2', 'JMJD1C',
		'KANK1', 'KANSL1', 'KAT2B', 'KAT6A', 'KAT6B', 'KATNAL1', 'KATNAL2', 'KCNA2', 'KCNB1', 'KCNC1',
		'KCNC2', 'KCND2','KCND3', 'KCNH5', 'KCNJ10', 'KCNJ15', 'KCNK7', 'KCNMA1', 'KCNQ2', 'KCNQ3', 'KCNS3', 'KCTD13',
		'KDM1B', 'KDM2A', 'KDM2B', 'KDM3A', 'KDM3B', 'KDM4B', 'KDM4C', 'KDM5A', 'KDM5B', 'KCNA3',
		'KDM5C', 'KDM6A', 'KDM6B', 'KHDRBS2', 'KIAA0232', 'KIAA1586', 'KIF13B', 'KIF14', 'KIF1A',
		'KIF5C', 'KIRREL3', 'KIZ', 'KLF16', 'KLF7', 'KLHL20', 'KMT2A', 'KMT2C', 'KMT2E', 'KMT5B',
		'KNG1', 'KPTN', 'KRR1', 'KRT26', 'LAMA1', 'LAMB1', 'LAS1L', 'LDB1', 'LDLR', 'LEMD3', 'LEO1',
		'LEP', 'LHX2', 'LILRB2', 'LIN7B', 'LMTK3', 'LNPK', 'LRBA', 'LRFN2', 'LRFN5', 'LRP1', 'LRP2',
		'LRRC1', 'LRRC4', 'LRRC4C', 'LZTR1', 'MACF1', 'MACROD2', 'MAGEL2', 'MAOA', 'MAOB', 'MAP1A',
		'MAP1B', 'MAP4K1', 'MAP4K4', 'MAPK3', 'MAPK8IP1', 'MAPT', 'MARK1', 'MARK2', 'MAST3', 'MBD1',
		'MBD3', 'MBD4', 'MBD5', 'LMX1B', 'LZTS2', 'MAPT-AS1', 'MBD6', 'MBOAT7', 'MCM4', 'MCM6', 'MCPH1',
		'MDGA2', 'MECP2', 'MED12L', 'MED13', 'MED13L', 'MED23', 'MEF2C', 'MEGF10','MEGF11', 'MEIS2', 
		'MEMO1', 'MET', 'METTL26', 'MFRP', 'MIB1', 'MINK1', 'MIR137', 'MKX',
		'MLANA', 'MNT', 'MRTFB', 'MSANTD2', 'MSL2', 'MSL3', 'MSR1', 'MSRA', 'MSX2', 'MTF1', 'MTHFR',
		'MTOR', 'MTSS2', 'MUC12', 'MUC4', 'MYCBP2', 'MYH10', 'MYH4', 'MYH9', 'MYLK', 'MYO16', 'MYO1E',
		'MYO5A', 'MYO5C', 'MYO9B', 'MYOCD', 'MYT1L', 'NAA10', 'NAA15', 'NAALADL2', 'NACC1', 'NAV2', 'NAV3',
		'NBEA', 'NCAPH2', 'NCKAP1', 'NCKAP5', 'NCOA1', 'NCOR1', 'NEGR1', 'NEO1', 'NEXMIF', 'NF1', 'NFE2L3',
		'NFIA', 'NFIB', 'NFIX', 'NINL', 'NIPA1', 'NIPA2', 'NIPBL', 'NLGN1', 'NLGN2', 'NLGN3', 'NLGN4X',
		'NLGN4Y', 'NOVA2', 'NPFFR2', 'NR1D1', 'NR2F1', 'NR3C2', 'NR4A2', 'NRCAM', 'NRP2', 'NRXN1', 'NDUFA5',
		'NKX2-2', 'NPAS2', 'NRXN2', 'NRXN3', 'NSD1', 'MSNP1AS', 'NSD2', 'NSMCE3', 'NTNG1', 'NTNG2',
		'NTRK1', 'NTRK2', 'NTRK3', 'NUAK1', 'NUDCD2', 'NUP133', 'NUP155', 'NXF1', 'NXPH1', 'OCRL', 'OFD1',
		'OPHN1', 'OR1C1', 'OR2T10','OR52M1', 'OTUD7A', 'OXT', 'OXTR', 'P2RX5', 'P4HA2', 'PABPC1', 'PACS1', 'PACS2', 'PAFAH1B2',
		'PAH', 'PAK1', 'PAK2', 'PAPOLG', 'PARD3B', 'PATJ', 'PAX5', 'PAX6', 'PBX1', 'PC', 'PCCA', 'PCCB',
		'PCDH10', 'PCDH11X', 'PCDH15', 'PCDH19', 'PCDH9', 'PCDHA1', 'PCDHA10', 'PCDHA11', 'PCDHA12',
		'PCDHA13', 'PCDHA2', 'PCDHA3', 'PCDHA4', 'PCDHA5', 'PCDHA6', 'PCDHA7', 'PCDHA8', 'PCDHA9',
		'OR2M4', 'OTX1', 'PCDHAC1', 'PCDHAC2', 'PCLO', 'PCM1', 'PDCD1', 'PDE1C', 'PDE3B', 'PDK2',
		'PDZD8', 'PEBP4', 'PER1', 'PER2', 'PEX7', 'PHB1', 'PHF12', 'PHF14', 'PHF2', 'PHF21A', 'PHF3',
		'PHF7', 'PHF8', 'PHIP', 'PHLPP1', 'PHRF1', 'PIK3CA', 'PIK3CG', 'PIK3R2', 'PITX1', 'PJA1', 'PLAUR',
		'PLCB1', 'PLCD4', 'PLEKHA8', 'PLN', 'PLPPR4', 'PLXNA3', 'PLXNA4', 'PLXNB1', 'PNPLA7', 'POGZ',
		'POLA2', 'POLR2A', 'POLR3A', 'POMGNT1', 'POMT1', 'PON1', 'POT1', 'POU3F3', 'PPFIA1', 'PPM1D',
		'PPP1R1B', 'PPP1R9B', 'PPP2CA', 'PPP2R1B', 'PPP2R5D', 'PPP3CA', 'PPP5C', 'PREX1','PRICKLE1', 
		'PRICKLE2', 'PRKAR1B', 'PRKCA', 'PRKCB', 'PRKD1', 'PRKD2', 'PRKDC', 'PRKN', 'PRODH',
		'PRPF39', 'PRPF8', 'PRR12', 'PRR14L', 'PRR25', 'PRUNE2', 'PSD3', 'PSMD11', 'PSMD12', 'PSMD6',
		'PTBP2', 'PTCHD1', 'PTCHD1-AS', 'PTDSS1', 'PTEN', 'PTGS2', 'PTK7', 'PTPN11', 'PTPN4', 'PTPRB',
		'PTPRC', 'PTPRT', 'PXDN', 'PYHIN1', 'QRICH1', 'PRPF19', 'RAB11FIP5', 'RAB2A', 'RAB39B', 'RAB43',
		'RAC1', 'RAD21', 'RAD21L1', 'RAI1', 'RALA', 'RALGAPB', 'RANBP17', 'RAPGEF4', 'RASSF5', 'RBBP5',
		'RBFOX1', 'RBM27', 'REEP3', 'RELN', 'RERE', 'RFX3', 'RFX4', 'RFX7', 'RGS7', 'RHEB', 'RHOXF1',
		'RIMS1', 'RIMS2', 'RIMS3', 'RIT2', 'RLIM', 'RNF135', 'RNF25', 'RNF38', 'ROBO2', 'RORA', 'RORB',
		'RPH3A', 'RPL10', 'RP11-1407O15.2', 'RPS6KA2', 'RPS6KA3', 'RSRC1', 'RUNX1T1', 'SAE1', 'SAMD11',
		'SASH1', 'SATB1', 'SATB2', 'RPS10P2-AS1', 'SBF1', 'SCAF1', 'SCAF4', 'SCFD2', 'SCGN', 'SCN1A',
		'SCN2A', 'SACS', 'SCN4A', 'SCN8A', 'SCN9A', 'SCP2', 'SDC2', 'SEMA5A', 'SENP1','SET', 
		'SETBP1', 'SETD1A', 'SETD1B', 'SETD2', 'SETD5', 'SETDB1', 'SETDB2', 'SEZ6L2', 'SF3B1',
		'SGSH', 'SGSM3', 'SH3RF1', 'SH3RF3', 'SHANK1', 'SHANK2', 'SERPINE1', 'SHANK3', 'SHOX', 'SIK1',
		'SIN3A', 'SIN3B', 'SKI', 'SLC12A5', 'SLC1A1', 'SLC1A2', 'SLC22A9', 'SLC23A1', 'SLC24A2', 'SLC25A12',
		'SLC25A39', 'SLC27A4', 'SLC29A4', 'SLC35G1', 'SLC38A10', 'SLC45A1', 'SLC4A10', 'SLC6A1', 'SLC6A3',
		'SLC6A4', 'SLC6A8', 'SLC7A3', 'SLC7A5', 'SLC7A7', 'SLC9A6', 'SLC9A9', 'SLCO1B3', 'SLFN5', 'SLITRK2',
		'SLITRK5', 'SMAD4', 'SMARCA2', 'SMARCA4', 'SMARCC2', 'SMC1A', 'SMC3', 'SMG6', 'SMURF1', 'SNAP25',
		'SNCAIP', 'SND1', 'SNTG2', 'SNX14', 'SNX5', 'SOD1', 'SON', 'SORCS3', 'SOS2', 'SOX5', 'SOX6',
		'SPARCL1', 'SPAST', 'SPEN', 'SPP2', 'SPRY2', 'SPTBN1', 'SRCAP', 'SRGAP3', 'SRPRA', 'SRRM2', 'SRSF1',
		'SRSF11', 'SSRP1', 'ST7', 'ST8SIA2', 'STAG1', 'STX1A', 'STXBP1', 'STXBP5', 'STYK1', 'SUPT16H',
		'SYAP1', 'SYBU', 'SLC22A15', 'SLC25A27', 'SLC35B1', 'STK39', 'SYCE1', 'SYN1', 'SYN2', 'SYNCRIP', 
		'SYNE1', 'SYNGAP1', 'SYNJ1', 'SYP', 'SYT1', 'TAF1', 'TAF1C', 'TAF4', 'TAF6', 'TANC2',
		'TAOK1', 'TAOK2', 'TBC1D23', 'TBC1D31', 'TBC1D5', 'TBCEL', 'TBCK', 'TBL1XR1', 'TBR1', 'TBX1',
		'TBX22', 'TCEAL1', 'TCF20', 'TCF4', 'TCF7L2', 'TECTA', 'TEK', 'TERB2', 'TERF2', 'TET2', 'TET3',
		'TFB2M', 'TFE3', 'TGM1', 'THBS1', 'THRA', 'TLE3', 'TLK2', 'TM4SF19', 'TM4SF20', 'TM9SF4', 'TMEM134',
		'TMEM39B', 'TMLHE', 'TNPO3', 'TNRC6B', 'TNRC6C', 'TNS2', 'TOP2B', 'TOP3B', 'TRAF7', 'TRAPPC2L',
		'TRAPPC6B', 'TRAPPC9', 'TRIM23', 'TRIM32', 'TRIM33', 'TRIM8', 'TRIO', 'TRIP12', 'TRPC5', 'TRPC6',
		'TRPM1', 'TRPM3', 'TRPM6', 'TRRAP', 'TSC1', 'SYT17', 'TBL1X', 'TDO2', 'TPO', 'TSC2', 'TSHZ1',
		'TSHZ3', 'TSPAN17', 'TSPAN4', 'TSPAN7', 'TSPOAP1', 'TTI2', 'TTN', 'TUBGCP5', 'UBAP2L', 'UBE2H',
		'UBE3A', 'U2AF2', 'UBE3C', 'UBN2', 'UBR1', 'UBR5', 'UIMC1', 'UNC13A', 'UNC5D', 'UNC79', 'UNC80',
		'UPF2', 'UPF3B', 'USH2A', 'USP15', 'USP30','USP45', 'USP7', 'USP9X', 'USP9Y', 'VAMP2', 'VEZF1', 
		'VIL1', 'VPS13B', 'VPS54', 'VSIG4', 'VWA7','WAC', 'WASF1', 'WDFY3', 'WDFY4', 'WDR26', 
		'WDR5', 'WNK3', 'WNT1', 'WWOX', 'WWP1', 'XPC', 'XPO1',
		'XRCC6', 'YEATS2', 'YTHDC2', 'YWHAE', 'VASH1', 'VCP', 'VDR', 'WDR37', 'YWHAG', 'YWHAZ', 'YY1',
		'ZBTB16', 'ZBTB18', 'ZBTB20', 'ZBTB21', 'ZBTB7A', 'ZC3H11A', 'ZC3H4', 'ZFYVE26', 'ZBTB47', 'ZMIZ1',
		'ZMYM2', 'ZMYM3', 'ZMYND11', 'ZMYND8', 'ZNF18', 'ZNF292', 'ZNF385B', 'ZNF462', 'ZNF517', 'ZNF548',
		'ZNF559', 'ZNF626', 'ZNF711', 'ZNF713', 'ZNF774', 'ZNF804A', 'ZNF827', 'ZSWIM6', 'ZWILCH'
}


input_dir = '/home/ibab/SEM_4/project/data/gvcf/final/filt_ann_gvcf'
output_dir = '/home/ibab/SEM_4/project/data/gvcf/final/filt_ann_gvcf'
for filename in os.listdir(input_dir):
    if filename.endswith('.gvcf'):
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, f"{filename[:-5]}.tsv")  # Replace .gvcf with .tsv
        process_gvcf(input_file, output_file,associated_genes)

# Process all .gvcf files in the input directory
print(f"Processed VCF files in {input_dir} and wrote results to {output_dir}")
