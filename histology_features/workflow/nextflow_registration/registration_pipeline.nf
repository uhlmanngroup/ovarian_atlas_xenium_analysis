process submit_registration {
    
    publishDir "${params.outdir}", mode: 'copy', pattern: "*.zarr"
    debug true

    tag "$sample_id"
    
    input:
    tuple val(sample_id), val(params_map)
    
    output:
    tuple val(sample_id), val(params_map), path("${sample_id}_adjacent.zarr")

    script:
    """
    ~/.local/bin/histology_features submit-registration ${params_map.config_path}
    """
}

// process publish_outputs {
//     publishDir "${params.outdir}", mode: 'copy'
    
//     input:
//     tuple val(sample_id), val(params_map), path(files)
    
//     output:
//     path "*.zarr"
    
//     // script:
//     // // Use output_name from sample sheet if provided, otherwise fall back to sample_id
//     // output_name = params_map.output_name ?: "${sample_id}_adjacent.zarr"
//     // """
//     // cp -r ${files}/${files} ${output_name}
//     // """
// }

workflow {
    samples_ch = Channel
    .fromPath(params.samplesheet)
    .splitCsv(header: true, sep: '\t')
    .map { row -> 
        return [
            row.sample_id, 
            row,
        ]
    }

    submit_registration(samples_ch)
}