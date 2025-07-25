"""
Análisis de Bioactividad Molecular para Toxoplasma
==================================================
Este script realiza un análisis completo de datos de bioactividad molecular,
incluyendo cálculo de descriptores, transformaciones de datos, visualizaciones
y análisis estadístico.
"""

# =============================================================================
# IMPORTACIONES
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from numpy.random import seed

# RDKit para descriptores moleculares
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

# Configuración de visualización
sns.set(style='ticks')

# =============================================================================
# FUNCIONES PARA DESCRIPTORES MOLECULARES
# =============================================================================

def calculate_lipinski_descriptors(smiles_list, verbose=False):
    """
    Calcula los descriptores de la Regla de Lipinski para una lista de SMILES.
    
    Parameters:
    -----------
    smiles_list : list
        Lista de cadenas SMILES
    verbose : bool
        Si True, imprime información adicional
    
    Returns:
    --------
    pd.DataFrame
        DataFrame con los descriptores calculados
    """
    # Convertir SMILES a objetos moleculares
    molecules = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        molecules.append(mol)
    
    # Calcular descriptores
    descriptors_data = []
    for mol in molecules:
        if mol is not None:
            desc_data = {
                'MW': Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'NumHDonors': Lipinski.NumHDonors(mol),
                'NumHAcceptors': Lipinski.NumHAcceptors(mol)
            }
            descriptors_data.append(desc_data)
    
    return pd.DataFrame(descriptors_data)

# =============================================================================
# FUNCIONES PARA TRANSFORMACIÓN DE DATOS
# =============================================================================

def normalize_bioactivity_values(df, value_column='standard_value', max_value=100000000):
    """
    Normaliza los valores de bioactividad aplicando un límite máximo.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    value_column : str
        Nombre de la columna con valores a normalizar
    max_value : float
        Valor máximo a aplicar
    
    Returns:
    --------
    pd.DataFrame
        DataFrame con columna normalizada añadida
    """
    df_norm = df.copy()
    df_norm['standard_value_norm'] = df_norm[value_column].apply(lambda x: min(x, max_value))
    df_norm = df_norm.drop(value_column, axis=1)
    return df_norm

def convert_to_pic50(df, value_column='standard_value_norm'):
    """
    Convierte valores IC50 a pIC50.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    value_column : str
        Nombre de la columna con valores IC50
    
    Returns:
    --------
    pd.DataFrame
        DataFrame con columna pIC50 añadida
    """
    df_pic50 = df.copy()
    # Conversión: pIC50 = -log10(IC50 * 10^-9)
    df_pic50['pIC50'] = df_pic50[value_column].apply(lambda x: -np.log10(x * 10**-9))
    df_pic50 = df_pic50.drop(value_column, axis=1)
    return df_pic50

def filter_two_classes(df, class_column='bioactivity_class', exclude_class='intermediate'):
    """
    Filtra el DataFrame para mantener solo dos clases de bioactividad.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    class_column : str
        Nombre de la columna de clases
    exclude_class : str
        Clase a excluir
    
    Returns:
    --------
    pd.DataFrame
        DataFrame filtrado
    """
    return df[df[class_column] != exclude_class].copy()

# =============================================================================
# FUNCIONES PARA VISUALIZACIÓN
# =============================================================================

def plot_bioactivity_distribution(df, class_column='bioactivity_class', 
                                 figsize=(6.5, 6.5), save_path='plot_bioactivity_class.pdf'):
    """
    Crea un gráfico de barras de la distribución de clases de bioactividad.
    """
    plt.figure(figsize=figsize)
    sns.countplot(x=class_column, data=df, edgecolor='black')
    plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
    plt.ylabel('Frequency', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_mw_vs_logp_scatter(df, figsize=(22.5, 22.5), save_path='plot_MW_vs_LogP.pdf'):
    """
    Crea un scatter plot de MW vs LogP coloreado por bioactividad.
    """
    plt.figure(figsize=figsize)
    sns.scatterplot(x='MW', y='LogP', data=df, hue='bioactivity_class', 
                   size='pIC50', edgecolor='black', alpha=0.7)
    plt.xlabel('MW', fontsize=14, fontweight='bold')
    plt.ylabel('LogP', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_descriptor_boxplot(df, descriptor, class_column='bioactivity_class', 
                           figsize=(5.5, 5.5), save_path=None):
    """
    Crea un boxplot para un descriptor específico por clase de bioactividad.
    """
    plt.figure(figsize=figsize)
    sns.boxplot(x=class_column, y=descriptor, data=df)
    plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
    plt.ylabel(descriptor, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path is None:
        save_path = f'plot_{descriptor}.pdf'
    
    plt.savefig(save_path)
    plt.show()

# =============================================================================
# FUNCIONES PARA ANÁLISIS ESTADÍSTICO
# =============================================================================

def mann_whitney_test(df, descriptor, class_column='bioactivity_class', 
                     active_class='active', inactive_class='inactive', 
                     alpha=0.05, random_seed=1):
    """
    Realiza la prueba de Mann-Whitney U para comparar distribuciones entre clases.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    descriptor : str
        Nombre del descriptor a analizar
    class_column : str
        Nombre de la columna de clases
    active_class : str
        Nombre de la clase activa
    inactive_class : str
        Nombre de la clase inactiva
    alpha : float
        Nivel de significancia
    random_seed : int
        Semilla para reproducibilidad
    
    Returns:
    --------
    pd.DataFrame
        DataFrame con los resultados de la prueba
    """
    # Configurar semilla para reproducibilidad
    seed(random_seed)
    
    # Separar datos por clase
    active_data = df[df[class_column] == active_class][descriptor]
    inactive_data = df[df[class_column] == inactive_class][descriptor]
    
    # Realizar la prueba
    statistic, p_value = mannwhitneyu(active_data, inactive_data)
    
    # Interpretar resultados
    if p_value > alpha:
        interpretation = 'Same distribution (fail to reject H0)'
    else:
        interpretation = 'Different distribution (reject H0)'
    
    # Crear DataFrame con resultados
    results = pd.DataFrame({
        'Descriptor': [descriptor],
        'Statistics': [statistic],
        'p': [p_value],
        'alpha': [alpha],
        'Interpretation': [interpretation]
    })
    
    # Guardar resultados
    filename = f'mannwhitneyu_{descriptor}.csv'
    results.to_csv(filename, index=False)
    
    return results

# =============================================================================
# FUNCIÓN PRINCIPAL DE ANÁLISIS
# =============================================================================

def main_analysis():
    """
    Función principal que ejecuta todo el pipeline de análisis.
    """
    print("=== INICIANDO ANÁLISIS DE BIOACTIVIDAD MOLECULAR ===\n")
    
    # 1. Cargar datos
    print("1. Cargando datos...")
    df = pd.read_csv('archivo_convertido.csv')
    print(f"   Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas\n")
    
    # 2. Calcular descriptores de Lipinski
    print("2. Calculando descriptores moleculares...")
    df_lipinski = calculate_lipinski_descriptors(df['canonical_smiles'])
    df_combined = pd.concat([df, df_lipinski], axis=1)
    df_combined.to_csv('toxoplasma_05_bioactivity_data_curated.csv', index=False)
    print("   Descriptores calculados y datos guardados\n")
    
    # 3. Normalizar y convertir datos
    print("3. Procesando datos de bioactividad...")
    df_norm = normalize_bioactivity_values(df_combined)
    print(f"   Estadísticas de valores normalizados:\n{df_norm['standard_value_norm'].describe()}\n")
    df_norm.to_csv('toxoplasma_06_bioactivity_data_curated.csv', index=False)
    
    df_final = convert_to_pic50(df_norm)
    df_final.to_csv('toxoplasma_07_bioactivity_data_curated.csv', index=False)
    print(f"   Estadísticas de pIC50:\n{df_final['pIC50'].describe()}\n")
    
    # 4. Filtrar a dos clases
    print("4. Filtrando datos a dos clases...")
    df_2class = filter_two_classes(df_final)
    df_2class.to_csv('toxoplasma_08_bioactivity_data_curated.csv', index=False)
    print(f"   Datos filtrados: {df_2class.shape[0]} filas\n")
    
    # 5. Generar visualizaciones
    print("5. Generando visualizaciones...")
    plot_bioactivity_distribution(df_2class)
    plot_mw_vs_logp_scatter(df_2class)
    
    descriptors = ['pIC50', 'MW', 'LogP', 'NumHDonors', 'NumHAcceptors']
    for desc in descriptors:
        plot_descriptor_boxplot(df_2class, desc)
    print("   Visualizaciones completadas\n")
    
    # 6. Análisis estadístico
    print("6. Realizando análisis estadístico...")
    statistical_results = []
    for desc in ['MW', 'LogP', 'NumHDonors', 'NumHAcceptors']:
        result = mann_whitney_test(df_2class, desc)
        statistical_results.append(result)
        print(f"   {desc}: p-value = {result['p'].iloc[0]:.4f}")
    
    # Consolidar resultados estadísticos
    all_results = pd.concat(statistical_results, ignore_index=True)
    all_results.to_csv('all_mannwhitney_results.csv', index=False)
    
    print("\n=== ANÁLISIS COMPLETADO ===")
    print("Archivos generados:")
    print("- toxoplasma_05_bioactivity_data_curated.csv (con descriptores)")
    print("- toxoplasma_06_bioactivity_data_curated.csv (datos normalizados)")
    print("- toxoplasma_07_bioactivity_data_curated.csv (con pIC50)")
    print("- toxoplasma_08_bioactivity_data_curated.csv (dos clases)")
    print("- Múltiples gráficos PDF")
    print("- Resultados estadísticos CSV")

# =============================================================================
# EJECUCIÓN
# =============================================================================

if __name__ == "__main__":
    main_analysis()
