from bravado.client import SwaggerClient
import pandas as pd 


class CBioDb:
    def __init__(self) -> None:
        self.conn = SwaggerClient.from_url('https://www.cbioportal.org/api/v2/api-docs',
                                           config={"validate_requests":False,"validate_responses":False,"validate_swagger_spec":False})
        self._all_study_ids = None 
        self._all_studies = None 
        self._all_fields = None 

    @property
    def all_studies(self):
        if self._all_study_ids is None:
            return self.conn.Studies.getAllStudiesUsingGET().result()

    @property
    def all_fields(self):
        if self._all_fields is None:
            return dir(self.all_studies[0])
            
    @property
    def all_study_ids(self):
        if self._all_study_ids is None:
            study_ids = [x.studyId for x in self.all_studies]
            return study_ids
            
    def get_study(self, study_id):
        return self.conn.Studies.getStudyUsingGET(studyId=study_id).result()
    
    def get_clinical_data(self, study_id):
        return self.conn.Clinical_Data.getAllClinicalDataInStudyUsingGET(studyId=study_id).result()
    
    def get_case_lists(self, study_id):
        return self.conn.Case_Lists.getAllCaseListsInStudyUsingGET(studyId=study_id).result()
    
    def get_case_list(self, study_id, case_list_id):
        return self.conn.Case_Lists.getCaseListUsingGET(studyId=study_id, caseListId=case_list_id).result()
    
    def get_case_list_data(self, study_id, case_list_id):
        return self.conn.Case_Lists.getAllCasesInCaseListUsingGET(studyId=study_id, caseListId=case_list_id).result()
    
    def get_molecular_profiles(self, study_id):
        return self.conn.Molecular_Profiles.getAllMolecularProfilesInStudyUsingGET(studyId=study_id).result()
    
    def get_molecular_profile(self, study_id, molecular_profile_id):
        return self.conn.Molecular_Profiles.getMolecularProfileUsingGET(studyId=study_id, molecularProfileId=molecular_profile_id).result()
    
    def get_molecular_profile_data(self, study_id, molecular_profile_id):
        return self.conn.Molecular_Profiles.getAllDataInMolecularProfileUsingGET(studyId=study_id, molecularProfileId=molecular_profile_id).result()
    
    def get_genes(self, study_id):
        return self.conn.Genes.getAllGenesInStudyUsingGET(studyId=study_id).result()
    
    def get_gene(self, study_id, gene_id):
        return self.conn.Genes.getGeneUsingGET(studyId=study_id, geneId=gene_id).result()
    
    def get_gene_data(self, study_id, gene_id):
        return self.conn.Genes.getAllGeneDataUsingGET(studyId=study_id, geneId=gene_id).result()
    
    def get_all_patients(self, study_id):
        return self.conn.Patients.getAllPatientsInStudyUsingGET(studyId=study_id).result()
    
    def get_data(self, method_name, **kwargs):
        method = getattr(self.conn, method_name)
        return method(**kwargs).result()



class CBioDbToDataFrame:
    def __init__(self, cbio_db: CBioDb) -> None:
        self.cbio_db = cbio_db

    def get_all_studies(self):
        return pd.DataFrame(self.cbio_db.get_data('Studies.getAllStudiesUsingGET'))
    
    def get_study(self, study_id):
        return pd.DataFrame(self.cbio_db.get_data('Studies.getStudyUsingGET', studyId=study_id))

    def get_molecular_profiles(self, study_id):
        return pd.DataFrame(self.cbio_db.get_data('Molecular_Profiles.getAllMolecularProfilesInStudyUsingGET', studyId=study_id))

    def get_molecular_profile(self, study_id, molecular_profile_id):
        return pd.DataFrame(self.cbio_db.get_data('Molecular_Profiles.getMolecularProfileUsingGET', studyId=study_id, molecularProfileId=molecular_profile_id))

    def get_molecular_profile_data(self, study_id, molecular_profile_id):
        return pd.DataFrame(self.cbio_db.get_data('Molecular_Profiles.getAllDataInMolecularProfileUsingGET', studyId=study_id, molecularProfileId=molecular_profile_id))

    def get_genes(self, study_id):
        return pd.DataFrame(self.cbio_db.get_data('Genes.getAllGenesInStudyUsingGET', studyId=study_id))

    def get_gene(self, study_id, gene_id):
        return pd.DataFrame(self.cbio_db.get_data('Genes.getGeneUsingGET', studyId=study_id, geneId=gene_id))

    def get_gene_data(self, study_id, gene_id):
        return pd.DataFrame(self.cbio_db.get_data('Genes.getAllGeneDataUsingGET', studyId=study_id, geneId=gene_id))

    def get_all_patients(self, study_id):
        return pd.DataFrame(self.cbio_db.get_data('Patients.getAllPatientsInStudyUsingGET', studyId=study_id))
    
    def get_clinical_data(self, study_id):
        return pd.DataFrame(self.cbio_db.get_data('Clinical_Data.getAllClinicalDataInStudyUsingGET', studyId=study_id))
    
    ## 375 studies/400 studies - will be the comprehensive to cover the mapping of category query to ontology
    ## Input - new file (Excel spreadsheet) - different column names
    ## Task Specification of harmonizer 
    ## Sequential:
    ## Demo excel file (From Ritika) - template data file 

    # 1. Identify from the columns which are in the category of treatment related info, disease related info, bodysite (Not been done a lot)
    # Named Entity Recognition for column harmonization 
    # Bert-based classifier - 3 classes - Treatment, Disease, Bodysite
    # https://allenai.github.io/scispacy/ 

    # 2. Map the Values for the harmonized columns in the Excel spreadsheet to curated ontology 
    ## Task Specification of Recommender 
    # 1. Recommend the curated ontology based on the query
    ## Riddhika will be able to provide more context on the file context 
    ## For a curator - We want to solve 2 problems - 1) Harmonizer 2) Recommender
    ## User -> Study Name -> Clinical Data -> Harmonize 
    ## Metadata attribute - harmonized column name 

    ## Harmonize via column name or via list (check box option) - For both the curator and harmonizer (Ontology Mapping)
 
    ## Streamlit template from Michelle b   
