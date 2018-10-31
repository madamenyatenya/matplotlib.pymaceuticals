#
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Hide warning messages in notebook
import warnings
warnings.filterwarnings('ignore')

#-------------------------------------------------------
# File to Load (Remember to Change These)
mouse_drug_data_to_load = "data/mouse_drug_data.csv"
clinical_trial_data_to_load = "data/clinicaltrial_data.csv"

# Read the Mouse and Drug Data and the Clinical Trial Data
mouse_drug_data = pd.read_csv(mouse_drug_data_to_load)
clinical_trial_data = pd.read_csv(clinical_trial_data_to_load)
# Combine the data into a single dataset

Mouse_ID = clinical_trial_data['Mouse ID']
Drug = []
for ID in Mouse_ID:
    m_Drug = np.array(mouse_drug_data.loc[ mouse_drug_data['Mouse ID']==ID,'Drug'])
    Drug.append(m_Drug[0])

clinical_labels = list(clinical_trial_data)
mouse_drug_labels = list(mouse_drug_data)
merged_label = clinical_labels;
merged_label.append(mouse_drug_labels[1])

panda_Drug = pd.DataFrame(Drug,columns=['Drug'])
merged_data = pd.concat([clinical_trial_data, panda_Drug], axis=1,ignore_index=True )
merged_data.columns = merged_label

# Display the data table for preview
print('\n------------- Merged data ---------------------------------------\n')
print(merged_data.head())

# Store the Mean Tumor Volume Data Grouped by Drug and Timepoint
unique_Drug = np.unique(merged_data['Drug'])
unique_Timepoint = np.unique(merged_data['Timepoint'])

Mean_Tumor_data = []
for m_Drug in unique_Drug:
    for m_Timepoint in unique_Timepoint:
        values = np.array(merged_data.loc[(merged_data['Drug'] == m_Drug) & (merged_data['Timepoint'] == m_Timepoint), merged_label[2]])
        Mean_Tumor_data.append([m_Drug, m_Timepoint, np.mean(values)])

# Convert to DataFrame
Data_Mean_Tumor = pd.DataFrame(Mean_Tumor_data, columns=['Drug','Timepoint','Tumor Volume (mm3)'])
# Preview DataFrame
print('\n------------- Mean Tumor ---------------------------------------\n')
print(Data_Mean_Tumor)


# Store the Standard Error of Tumor Volumes Grouped by Drug and Timepoint
std_Tumor_data = []
for m_Drug in unique_Drug:
    for m_Timepoint in unique_Timepoint:
        values = np.array(merged_data.loc[(merged_data['Drug'] == m_Drug) & (merged_data['Timepoint'] == m_Timepoint), merged_label[2]])
        std_Tumor_data.append([m_Drug, m_Timepoint, np.std(values)/np.sqrt(len(values)-1)])

# Convert to DataFrame
Data_std_Tumor = pd.DataFrame(std_Tumor_data, columns=['Drug','Timepoint','Tumor Volume (mm3)'])
# Preview DataFrame
print('\n------------- std Tumor ---------------------------------------\n')
print(Data_std_Tumor.head())


# Minor Data Munging to Re-Format the Data Frames
Re_format_data = pd.DataFrame(unique_Timepoint, columns=['Timepoint'])
for m_Drug in unique_Drug:
    values = np.array(Data_Mean_Tumor.loc[(Data_Mean_Tumor['Drug'] == m_Drug) , 'Tumor Volume (mm3)'])
    Re_format_data[m_Drug] = values

# Preview that Reformatting worked
print('\n------------- Re_format ---------------------------------------\n')
print(Re_format_data.head())

# Generate the Plot (with Error Bars)
plt.errorbar(Re_format_data['Timepoint'], Re_format_data[unique_Drug[0]],yerr=Data_std_Tumor.loc[(Data_std_Tumor['Drug'] == unique_Drug[0]) , 'Tumor Volume (mm3)'], fmt="ro:")
plt.errorbar(Re_format_data['Timepoint'], Re_format_data[unique_Drug[2]],yerr=Data_std_Tumor.loc[(Data_std_Tumor['Drug'] == unique_Drug[2]) , 'Tumor Volume (mm3)'], fmt="b^:")
plt.errorbar(Re_format_data['Timepoint'], Re_format_data[unique_Drug[3]],yerr=Data_std_Tumor.loc[(Data_std_Tumor['Drug'] == unique_Drug[3]) , 'Tumor Volume (mm3)'], fmt="gs:")
plt.errorbar(Re_format_data['Timepoint'], Re_format_data[unique_Drug[5]],yerr=Data_std_Tumor.loc[(Data_std_Tumor['Drug'] == unique_Drug[5]) , 'Tumor Volume (mm3)'], fmt="cd:")
plt.legend([unique_Drug[0],unique_Drug[2],unique_Drug[3],unique_Drug[5]],loc='upper left')
plt.grid()
plt.xlabel('Time(days)')
plt.ylabel('Tumor Volume(mm3)')
plt.title('Tumor Response to Treatment')
# Save the Figure
plt.savefig("Fig1.png")
plt.show()

# Store the Mean Met. Site Data Grouped by Drug and Timepoint
Mean_Met_data = []
for m_Drug in unique_Drug:
    for m_Timepoint in unique_Timepoint:
        values = np.array(merged_data.loc[(merged_data['Drug'] == m_Drug) & (merged_data['Timepoint'] == m_Timepoint), merged_label[3]])
        Mean_Met_data.append([m_Drug, m_Timepoint, np.mean(values)])

# Convert to DataFrame
Data_Mean_Met = pd.DataFrame(Mean_Met_data, columns=['Drug','Timepoint',merged_label[3]])

# Preview DataFrame
print('\n------------- Mean Met ---------------------------------------\n')
print(Data_Mean_Met.head())

# Store the Standard Error associated with Met. Sites Grouped by Drug and Timepoint
std_Met_data = []
for m_Drug in unique_Drug:
    for m_Timepoint in unique_Timepoint:
        values = np.array(merged_data.loc[(merged_data['Drug'] == m_Drug) & (merged_data['Timepoint'] == m_Timepoint), merged_label[3]])
        std_Met_data.append([m_Drug, m_Timepoint, np.std(values)/np.sqrt(len(values)-1)])

# Convert to DataFrame
Data_std_Met = pd.DataFrame(std_Met_data, columns=['Drug','Timepoint',merged_label[3]])
# Preview DataFrame
print('\n------------- std Met ---------------------------------------\n')
print(Data_std_Met.head())

# Minor Data Munging to Re-Format the Data Frames
Met_Re_format_data = pd.DataFrame(unique_Timepoint, columns=['Timepoint'])
for m_Drug in unique_Drug:
    values = np.array(Data_Mean_Met.loc[(Data_Mean_Met['Drug'] == m_Drug), merged_label[3]])
    Met_Re_format_data[m_Drug] = values

# Preview that Reformatting worked
print('\n------------- Met_Re_format ---------------------------------------\n')
print(Met_Re_format_data.head())


# Generate the Plot (with Error Bars)
plt.errorbar(Met_Re_format_data['Timepoint'], Met_Re_format_data[unique_Drug[0]],yerr=Data_std_Met.loc[(Data_std_Met['Drug'] == unique_Drug[0]) , merged_label[3]], fmt="ro:")
plt.errorbar(Met_Re_format_data['Timepoint'], Met_Re_format_data[unique_Drug[2]],yerr=Data_std_Met.loc[(Data_std_Met['Drug'] == unique_Drug[2]) , merged_label[3]], fmt="b^:")
plt.errorbar(Met_Re_format_data['Timepoint'], Met_Re_format_data[unique_Drug[3]],yerr=Data_std_Met.loc[(Data_std_Met['Drug'] == unique_Drug[3]) , merged_label[3]], fmt="gs:")
plt.errorbar(Met_Re_format_data['Timepoint'], Met_Re_format_data[unique_Drug[5]],yerr=Data_std_Met.loc[(Data_std_Met['Drug'] == unique_Drug[5]) , merged_label[3]], fmt="cd:")
plt.legend([unique_Drug[0],unique_Drug[2],unique_Drug[3],unique_Drug[5]],loc='upper left')
plt.grid()
plt.xlabel('Treatment Duration(Days)')
plt.ylabel(merged_label[3])
plt.title('Metastatic Spreed During Treatment')
# Save the Figure
plt.savefig("Fig2.png")
plt.show()

# Store the Count of Mice Grouped by Drug and Timepoint (W can pass any metric)
Mouse_count_data = []
for m_Drug in unique_Drug:
    for m_Timepoint in unique_Timepoint:
        values = np.array(merged_data.loc[(merged_data['Drug'] == m_Drug) & (merged_data['Timepoint'] == m_Timepoint), merged_label[2]])
        Mouse_count_data.append([m_Drug, m_Timepoint, len(values)])

# Convert to DataFrame
Data_Mouse_count = pd.DataFrame(Mouse_count_data, columns=['Drug','Timepoint','Mouse Count'])

# Preview DataFrame
print('\n------------- Mouse_count ---------------------------------------\n')
print(Data_Mouse_count.head())


# Minor Data Munging to Re-Format the Data Frames
Mouse_Re_format = pd.DataFrame(unique_Timepoint, columns=['Timepoint'])
for m_Drug in unique_Drug:
    values = np.array(Data_Mouse_count.loc[(Data_Mean_Met['Drug'] == m_Drug), 'Mouse Count'])
    Mouse_Re_format[m_Drug] = values

# Preview that Reformatting worked
print('\n------------- Mouse Mouse Re_format ---------------------------------------\n')
print(Mouse_Re_format.head())

# Generate the Plot (Accounting for percentages)
plt.errorbar(Mouse_Re_format['Timepoint'], Mouse_Re_format[unique_Drug[0]]*100.0/np.max(np.array(Mouse_Re_format[unique_Drug[0]])), fmt="ro:")
plt.errorbar(Mouse_Re_format['Timepoint'], Mouse_Re_format[unique_Drug[2]]*100.0/np.max(np.array(Mouse_Re_format[unique_Drug[2]])), fmt="b^:")
plt.errorbar(Mouse_Re_format['Timepoint'], Mouse_Re_format[unique_Drug[3]]*100.0/np.max(np.array(Mouse_Re_format[unique_Drug[3]])), fmt="gs:")
plt.errorbar(Mouse_Re_format['Timepoint'], Mouse_Re_format[unique_Drug[5]]*100.0/np.max(np.array(Mouse_Re_format[unique_Drug[5]])), fmt="cd:")
plt.legend([unique_Drug[0],unique_Drug[2],unique_Drug[3],unique_Drug[5]],loc='lower left')
plt.grid()
plt.xlabel('Time(Days)')
plt.ylabel('Survival Rate(%)')
plt.title('Survivla During Treatment')
# Save the Figure
plt.savefig("Fig3.png")
# Show the Figure
plt.show()

# Calculate the percent changes for each drug
percent_change = []
for m_Drug in unique_Drug:
    values = np.array(Data_Mean_Tumor.loc[(Data_Mean_Tumor['Drug'] == m_Drug) , 'Tumor Volume (mm3)'])
    percent_change.append((values[-1]-values[0])*100.0/values[0])

Tumor_change_data = pd.DataFrame(columns=('Drug','change_percent'))
Tumor_change_data.Drug = unique_Drug;
Tumor_change_data.change_percent = percent_change;

# Display the data to confirm
print('\n------------- Tumor_change_data ---------------------------------------\n')
print(Tumor_change_data)


# Store all Relevant Percent Changes into a Tuple
list_Tumor_change_data = np.array(Tumor_change_data)

# Splice the data between passing and failing drugs
list_Tumor_change_data1 = list_Tumor_change_data[[0,2,3,5],:]
# Use functions to label the percentages of changes
# Orient widths. Add labels, tick marks, etc.
def plot_graph(ax, list_Tumor_change_data1):
    for n in range(len(list_Tumor_change_data1)):
        if list_Tumor_change_data1[n,1]<0:
            ax.bar(n+0.5, list_Tumor_change_data1[n, 1], width=1, color='g')
            ax.text(n+0.3, -3, str(np.round(list_Tumor_change_data1[n, 1]))+"%", size=10,color="w")
        else:
            ax.bar(n+0.5, list_Tumor_change_data1[n, 1], width=1, color='r')
            ax.text(n + 0.3, 3, str(np.round(list_Tumor_change_data1[n, 1])) + "%", size=10,color="w")

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(list_Tumor_change_data1[:, 0])
    ax.grid()
    plt.ylabel('% Tumor Volnume Change')
    plt.title('Tumor Change Over 45 Day Treatment')

    return ax

# Call functions to implement the function calls
fig,ax = plt.subplots()
ax = plot_graph(ax, list_Tumor_change_data1)

# Save the Figure
plt.savefig("Fig4.png")

# Show the Figure
fig.show()
